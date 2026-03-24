"""
api_gpt.py — GPT-4o Vision OCR 모듈
MSDS 이미지에서 2절(유해성위험성 가목) + 3절(구성성분 + 함량 파싱) 추출

[함량 파싱 규칙]
  원문           → 최소    최대     미만여부
  "4~11%"        → 4.0     11.0     false
  "0.1~1미만"    → 0.1     0.99     true   ← -0.01
  "1%미만"       → 0.0     0.99     true   ← -0.01
  "<5%"          → 0.0     4.99     true   ← -0.01
  "≥1%"          → 1.0     null     false
  "1% 초과"      → 1.01    null     false  ← +0.01
  "1~5% 초과"    → 1.0     5.01     false  ← +0.01 (상한이 초과)
  "약 10%"       → null    10.0     false
"""

import base64, io, json, os, re, requests

# ── API 키는 환경변수에서 로드 ──────────────────────────────
# Render 대시보드 → Environment → OPENAI_KEY 에 값 입력
# 로컬 실행 시: 터미널에서 export OPENAI_KEY="sk-proj-..." 후 python app.py
OPENAI_KEY = os.environ.get("OPENAI_KEY", "")

MSDS_PROMPT = """
You are a document data extraction assistant. Extract structured information from the attached document image and return it as a JSON object.

This document is an official Korean regulatory compliance form (안전보건자료) used for workplace safety management under Korean occupational health law. Extracting its contents is required for legal compliance reporting.

Return ONLY a valid JSON object. No markdown code blocks, no explanation text before or after.
For missing fields use "해당없음". For unreadable fields use "".

{
  "section2": {
    "ghs_폭발성물질":       "",
    "ghs_인화성가스":       "",
    "ghs_에어로졸":         "",
    "ghs_산화성가스":       "",
    "ghs_고압가스":         "",
    "ghs_인화성액체":       "",
    "ghs_인화성고체":       "",
    "ghs_자기반응성물질":   "",
    "ghs_자연발화성액체":   "",
    "ghs_자연발화성고체":   "",
    "ghs_자기발열성물질":   "",
    "ghs_물반응성물질":     "",
    "ghs_산화성액체":       "",
    "ghs_산화성고체":       "",
    "ghs_유기과산화물":     "",
    "ghs_금속부식성물질":   ""
  },
  "section3": [
    {
      "cas_no": "",
      "ke_no": "",
      "함량원문": "",
      "함량최소": null,
      "함량최대": null,
      "최대포함여부": true,
      "최소포함여부": true
    }
  ]
}

SECTION 2 RULES:
- Find the table or list labeled "가. 물리적 위험성" inside Section 2 of this document.
- For each of the 16 ghs_ fields, copy the classification text exactly (e.g. "인화성 액체 구분 3").
- Write "해당없음" for any field not mentioned. Do NOT copy health or environmental hazard rows.

SECTION 3 RULES:
- Find the ingredient composition table in Section 3 (구성성분의 명칭 및 함유량).
- Create one JSON object per CAS number. Do NOT include 화학물질명 — only CAS and concentration.
- cas_no: ONE CAS number per JSON object (format: 000-00-0).
- ke_no: KE number if present (format: KE-00000), else "".
- 함량원문: concentration exactly as printed.

IMPORTANT — MULTIPLE CAS NUMBERS SHARING ONE CONCENTRATION:
If a single table row or cell group contains MULTIPLE CAS numbers but only ONE concentration value,
create a SEPARATE JSON object for EACH CAS number, all sharing the same concentration fields.
Example:
  Row has CAS: 96-49-1, 108-32-7, 105-58-8, 616-38-6 and concentration "80 - 90"
  → Output 4 separate objects, each with the same 함량원문 "80 - 90"

Also apply this rule when CAS numbers are listed as sub-items or in separate lines within one merged cell.

- 함량최소 (number or null): lower bound of concentration range.
    "4~11%" → 4,  "0.1~1미만" → 0.1,  "1%미만" → 0,  "<5%" → 0,  "≥1%" → 1,  "1%초과" → 1.01,  "10 - 20" → 10
- 함량최대 (number or null): upper bound. Use the EXACT printed number — do NOT subtract or add anything.
    "4~11%" → 11,  "0.1~1미만" → 1,  "1%미만" → 1,  "<5%" → 5,  "≥1%" → null,  "1%초과" → null,  "10 - 20" → 20
- 최대포함여부 (boolean): true if the upper bound VALUE IS INCLUDED (이하/≤/~까지/일반범위), false if NOT included (미만/<)
    "4~11%"     → true   (일반 범위: 11% 포함)
    "10 - 20"   → true   (일반 범위: 포함)
    "0.1~1미만" → false  (미만: 1% 미포함)
    "1%미만"    → false  (미만: 미포함)
    "<5%"       → false  (미포함)
    "≤5%"       → true   (이하: 포함)
    "5%이하"    → true   (이하: 포함)
    "≥1%"       → null   (최대값 없으므로 해당없음)
- 최소포함여부 (boolean): true if the lower bound VALUE IS INCLUDED (이상/≥/일반범위), false if NOT included (초과/>)
    "4~11%"  → true   (이상: 포함)
    "10 - 20" → true  (일반 범위: 포함)
    "1%초과" → false  (초과: 미포함)
    ">1%"    → false  (미포함)
    "≥1%"   → true   (이상: 포함)
"""

GHS_A_KEYS = [
    ("ghs_폭발성물질",      "폭발성 물질"),
    ("ghs_인화성가스",      "인화성 가스"),
    ("ghs_에어로졸",        "에어로졸"),
    ("ghs_산화성가스",      "산화성 가스"),
    ("ghs_고압가스",        "고압가스"),
    ("ghs_인화성액체",      "인화성 액체"),
    ("ghs_인화성고체",      "인화성 고체"),
    ("ghs_자기반응성물질",  "자기반응성 물질"),
    ("ghs_자연발화성액체",  "자연발화성 액체"),
    ("ghs_자연발화성고체",  "자연발화성 고체"),
    ("ghs_자기발열성물질",  "자기발열성 물질"),
    ("ghs_물반응성물질",    "물반응성 물질"),
    ("ghs_산화성액체",      "산화성 액체"),
    ("ghs_산화성고체",      "산화성 고체"),
    ("ghs_유기과산화물",    "유기과산화물"),
    ("ghs_금속부식성물질",  "금속 부식성 물질"),
]


def extract_pdf_text(file_bytes: bytes) -> str:
    """PDF 전체 텍스트 추출 (텍스트형 PDF 전용)."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"=== PAGE {i+1} ===\n{text}")
        return "\n\n".join(pages_text)
    except ImportError:
        raise RuntimeError("pypdf 미설치. 실행: pip install pypdf")


def pdf_to_images_b64(file_bytes: bytes, max_pages: int = 10) -> list[dict]:
    """
    스캔본 PDF → 각 페이지를 JPEG base64로 변환.
    반환: [{"b64": "...", "media_type": "image/jpeg"}, ...]
    """
    try:
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(
            file_bytes, dpi=150,
            first_page=1, last_page=max_pages,
        )
        result = []
        for page in pages:
            buf = io.BytesIO()
            page.save(buf, format="JPEG", quality=80)
            result.append({
                "b64":        base64.b64encode(buf.getvalue()).decode(),
                "media_type": "image/jpeg",
            })
        return result
    except ImportError:
        raise RuntimeError(
            "스캔본 PDF 처리에는 pdf2image + Poppler가 필요합니다.\n"
            "  pip install pdf2image\n"
            "  Poppler: https://github.com/oschwartz10612/poppler-windows/releases"
        )


def img_bytes_to_b64(file_bytes: bytes, filename: str) -> dict:
    """이미지 파일 → base64 dict."""
    media_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    return {"b64": base64.b64encode(file_bytes).decode(), "media_type": media_type}


def _call_gpt(user_content: list, headers: dict) -> str:
    """GPT-4o 호출 공통 함수. user_content 그대로 전달."""
    payload = {
        "model":      "gpt-4o",
        "max_tokens": 4000,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a data extraction assistant for Korean occupational safety compliance. "
                    "You extract structured data from official regulatory documents (MSDS). "
                    "You always respond with valid JSON only, following the user's schema exactly. "
                    "Never refuse to process official safety documentation."
                )
            },
            {"role": "user", "content": user_content},
        ],
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers, json=payload, timeout=120,
    )
    resp.raise_for_status()
    resp_json     = resp.json()
    finish_reason = resp_json["choices"][0].get("finish_reason", "")
    ai_text       = resp_json["choices"][0]["message"]["content"] or ""
    clean         = re.sub(r"```(?:json)?|```", "", ai_text).strip()

    if not clean:
        raise RuntimeError(
            f"GPT 응답 비어있음 (finish_reason={finish_reason})\n"
            f"원문: {ai_text[:200]!r}"
        )
    return clean


def _build_image_user_content(images: list[dict]) -> list:
    """
    이미지 목록 → GPT user content 블록.
    GPT-4o는 한 번에 최대 ~10장 처리 가능.
    """
    content = [{"type": "text", "text": MSDS_PROMPT}]
    for i, img in enumerate(images):
        content.append({
            "type": "image_url",
            "image_url": {
                "url":    f"data:{img['media_type']};base64,{img['b64']}",
                "detail": "high" if len(images) <= 4 else "low",
            }
        })
    return content


def _merge_section3(results: list[dict]) -> list:
    """
    여러 GPT 응답의 section3 배열을 합치고 중복 제거.
    중복 키 = (CAS번호 + 함량원문) 조합 — 같은 CAS라도 함량이 다르면 별도 행 유지.
    """
    seen = set()
    merged = []
    for r in results:
        for comp in r.get("section3", []):
            cas = comp.get("cas_no", "").strip()
            raw = comp.get("함량원문", "").strip()
            if cas and cas != "해당없음":
                key = f"{cas}||{raw}"
            else:
                key = str(id(comp))  # CAS 없으면 객체 id로 유일성 보장
            if key not in seen:
                seen.add(key)
                merged.append(comp)
    return merged


def _merge_section2(results: list[dict]) -> dict:
    """여러 GPT 응답의 section2 합치기 — 해당없음보다 실제 값 우선."""
    merged = {}
    for r in results:
        for k, v in r.get("section2", {}).items():
            if k not in merged or (merged[k] in ("해당없음", "", None) and v not in ("해당없음", "", None)):
                merged[k] = v
    return merged


def ocr_msds(file_bytes_list: list[tuple[bytes, str]]) -> dict:
    """
    MSDS 파일(들) → GPT-4o → {section2, section3}

    file_bytes_list: [(파일바이트, 파일명), ...]
      - PDF 1개: 텍스트형이면 텍스트 추출, 스캔본이면 페이지별 이미지
      - JPG/PNG 여러 장: 순서대로 GPT에 전달 (최대 10장씩 분할)

    반환: {section2: {...}, section3: [...]}
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type":  "application/json",
    }

    all_results = []

    for file_bytes, filename in file_bytes_list:
        fname = filename.lower()

        if fname.endswith(".pdf"):
            # ── PDF 처리 ───────────────────────────────
            pdf_text = extract_pdf_text(file_bytes)

            if pdf_text.strip():
                # 텍스트형 PDF → 텍스트로 GPT 전달
                user_content = [
                    {"type": "text", "text": MSDS_PROMPT},
                    {"type": "text", "text": f"\n\n[MSDS 문서 전체 텍스트]\n\n{pdf_text}"},
                ]
                clean = _call_gpt(user_content, headers)
                try:
                    all_results.append(json.loads(clean))
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"JSON 파싱 실패: {e}\n응답: {clean[:400]}")

            else:
                # 스캔본 PDF → 페이지별 이미지 변환
                images = pdf_to_images_b64(file_bytes, max_pages=10)
                # 4장씩 묶어서 GPT 호출 (토큰 한도)
                chunk_size = 4
                for i in range(0, len(images), chunk_size):
                    chunk   = images[i:i+chunk_size]
                    content = _build_image_user_content(chunk)
                    clean   = _call_gpt(content, headers)
                    try:
                        all_results.append(json.loads(clean))
                    except json.JSONDecodeError as e:
                        raise RuntimeError(f"JSON 파싱 실패 (페이지 {i+1}~): {e}\n응답: {clean[:400]}")

        else:
            # ── JPG / PNG 처리 ─────────────────────────
            img = img_bytes_to_b64(file_bytes, filename)
            content = _build_image_user_content([img])
            clean   = _call_gpt(content, headers)
            try:
                all_results.append(json.loads(clean))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON 파싱 실패: {e}\n응답: {clean[:400]}")

    if not all_results:
        raise RuntimeError("처리된 파일이 없습니다.")

    # ── 파일별 section3 인덱스 추적 (다중 파일 구분용) ──
    # all_results[i]는 file_bytes_list[i]에 대응
    per_file = []
    global_idx = 0
    seen_cas_global = set()

    for file_result, (_, fname) in zip(all_results, file_bytes_list):
        file_indices = []
        for comp in file_result.get("section3", []):
            cas = comp.get("cas_no", "").strip()
            key = cas if cas and cas != "해당없음" else id(comp)
            if key not in seen_cas_global:
                seen_cas_global.add(key)
                file_indices.append(global_idx)
                global_idx += 1
        per_file.append({"filename": fname, "section3_indices": file_indices})

    # 여러 결과 합치기
    result = {
        "section2":  _merge_section2(all_results),
        "section3":  _merge_section3(all_results),
        "per_file":  per_file if len(file_bytes_list) > 1 else [],
    }

    # ── 함량 후처리 ──────────────────────────────────
    for comp in result["section3"]:
        if "함량" in comp and "함량원문" not in comp:
            comp["함량원문"] = comp.pop("함량")
        # 구버전 호환: 미만여부/초과여부 → 최대포함여부/최소포함여부 변환
        if "미만여부" in comp and "최대포함여부" not in comp:
            comp["최대포함여부"] = not comp.pop("미만여부")
        if "초과여부" in comp and "최소포함여부" not in comp:
            comp["최소포함여부"] = not comp.pop("초과여부")
        raw = comp.get("함량원문", "")
        if comp.get("함량최대") is None and comp.get("함량최소") is None:
            comp.update(parse_content(raw))

    return result


def parse_content(raw: str) -> dict:
    """
    함량 원문 → {함량최소, 함량최대, 최대포함여부, 최소포함여부}
    
    최대포함여부: True = 이하(≤, 포함), False = 미만(<, 미포함)
    최소포함여부: True = 이상(≥, 포함), False = 초과(>, 미포함)
    """
    if not raw:
        return {"함량최소": None, "함량최대": None, "최대포함여부": True, "최소포함여부": True}

    s = raw.strip().replace(" ", "").replace("%", "").replace("％", "")

    # 최대값 포함 여부: 미만/<이면 False(미포함), 이하/≤이면 True(포함)
    is_lt   = bool(re.search(r"미만|<|＜", s))   # 미포함
    is_lte  = bool(re.search(r"이하|≤|<=", s))   # 포함
    # 최소값 포함 여부: 초과/>이면 False(미포함), 이상/≥이면 True(포함)
    is_gt   = bool(re.search(r"초과|>|＞", s))   # 미포함
    is_gte  = bool(re.search(r"이상|≥|>=", s))   # 포함

    max_included = (not is_lt) if (is_lt or is_lte) else True
    min_included = (not is_gt) if (is_gt or is_gte) else True

    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return {"함량최소": None, "함량최대": None, "최대포함여부": True, "최소포함여부": True}

    if len(nums) == 1:
        n = nums[0]
        if is_lt or is_lte:
            return {"함량최소": 0.0, "함량최대": n, "최대포함여부": max_included, "최소포함여부": True}
        if is_gt or is_gte:
            return {"함량최소": n, "함량최대": None, "최대포함여부": True, "최소포함여부": min_included}
        return {"함량최소": None, "함량최대": n, "최대포함여부": True, "최소포함여부": True}

    lo, hi = min(nums), max(nums)
    return {"함량최소": lo, "함량최대": hi, "최대포함여부": max_included, "최소포함여부": min_included}

