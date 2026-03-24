"""
app.py — MSDS AI OCR 분석 서버 (버전 3 · Railway 배포용)

엔드포인트:
  POST /api/msds/analyze   ← MSDS 파일 → GPT OCR → K-REACH · KOSHA
  GET  /api/chem           ← K-REACH 단독 검색
  GET  /api/kosha/search   ← KOSHA 단독 검색 (진단용)
  GET  /api/debug          ← 진단 정보
  GET  /                   ← 서버 상태
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import concurrent.futures, traceback, os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# ── 모듈 임포트 ──────────────────────────────────────
_import_errors = {}

try:
    from api_gpt import ocr_msds, GHS_A_KEYS, OPENAI_KEY
except Exception as e:
    _import_errors["api_gpt"] = str(e)
    ocr_msds = None; GHS_A_KEYS = []; OPENAI_KEY = ""

try:
    from api_kreach import search_by_cas as kreach_search, KREACH_URL, KREACH_KEY
except Exception as e:
    _import_errors["api_kreach"] = str(e)
    kreach_search = None; KREACH_URL = ""; KREACH_KEY = ""

try:
    from api_kosha import search_by_cas as kosha_search, KOSHA_URL, KOSHA_KEY
except Exception as e:
    _import_errors["api_kosha"] = str(e)
    kosha_search = None; KOSHA_URL = ""; KOSHA_KEY = ""

try:
    import requests as _req
except Exception as e:
    _import_errors["requests"] = str(e)
    _req = None


# ── 전역 에러 핸들러 ──────────────────────────────────
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e), "trace": traceback.format_exc()[-800:]}), 500

@app.errorhandler(404)
def not_found(e):
    # 프론트엔드 라우팅 — index.html 반환
    try:
        return send_from_directory('.', 'index.html')
    except Exception:
        return jsonify({"error": f"엔드포인트 없음: {request.path}"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": f"허용되지 않은 메서드: {request.method}"}), 405


# ══════════════════════════════════════════════
#  상태 확인
# ══════════════════════════════════════════════
@app.route("/")
def index():
    openai_ok = bool(OPENAI_KEY) and OPENAI_KEY.startswith("sk-") and len(OPENAI_KEY) > 20
    status = {
        "status": "ok",
        "services": {
            "msds_ocr": "준비됨" if (openai_ok and not _import_errors.get("api_gpt"))
                        else f"⚠️ {_import_errors.get('api_gpt', 'OPENAI_KEY 미설정')}",
            "kreach":   "정상" if not _import_errors.get("api_kreach") else f"⚠️ {_import_errors['api_kreach']}",
            "kosha":    "정상" if not _import_errors.get("api_kosha")  else f"⚠️ {_import_errors['api_kosha']}",
        }
    }
    if _import_errors:
        status["import_errors"] = _import_errors
    return jsonify(status)


@app.route("/api/debug")
def debug():
    return jsonify({
        "python":         __import__("sys").version,
        "cwd":            os.getcwd(),
        "import_errors":  _import_errors,
        "openai_key_set": bool(OPENAI_KEY) and OPENAI_KEY.startswith("sk-"),
        "kreach_url":     KREACH_URL,
        "kosha_url":      KOSHA_URL,
    })


# ══════════════════════════════════════════════
#  MSDS 통합 분석 (GPT OCR → K-REACH · KOSHA)
# ══════════════════════════════════════════════
@app.route("/api/msds/analyze", methods=["POST"])
def msds_analyze():
    if ocr_msds is None:
        return jsonify({"error": f"api_gpt 로드 실패: {_import_errors.get('api_gpt','')}"}), 500
    if kreach_search is None:
        return jsonify({"error": f"api_kreach 로드 실패: {_import_errors.get('api_kreach','')}"}), 500

    # 파일 수집
    uploaded = []
    if "file" in request.files:
        f = request.files["file"]
        uploaded.append((f.read(), f.filename))
    elif "files[]" in request.files:
        for f in request.files.getlist("files[]"):
            uploaded.append((f.read(), f.filename))
    else:
        return jsonify({"error": "파일 없음. 필드명: 'file' 또는 'files[]'"}), 400

    if not uploaded:
        return jsonify({"error": "파일이 비어있습니다."}), 400

    def enrich(comp: dict) -> dict:
        """구성성분 1개에 K-REACH · KOSHA 조회 결과 추가."""
        cas   = comp.get("cas_no", "").strip()
        c_max = comp.get("함량최대")
        is_lt = not bool(comp.get("최대포함여부", True))
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                f_kr = ex.submit(kreach_search, cas, c_max, is_lt)
                f_ko = ex.submit(kosha_search, cas) if kosha_search else None
                comp["kreach"] = f_kr.result()
                comp["kosha"]  = f_ko.result() if f_ko else {"error": "KOSHA 모듈 없음"}
        except Exception as e:
            comp["kreach"] = {"error": str(e)}
            comp["kosha"]  = {"error": str(e)}
        return comp

    results = []
    for file_bytes, filename in uploaded:
        try:
            ocr = ocr_msds([(file_bytes, filename)])
        except Exception as e:
            results.append({
                "filename": filename,
                "error":    f"OCR 실패: {e}",
                "trace":    traceback.format_exc()[-400:],
            })
            continue

        section3 = [enrich(c) for c in ocr.get("section3", [])]
        results.append({
            "filename": filename,
            "section2": ocr.get("section2", {}),
            "section3": section3,
        })

    return jsonify({"ok": True, "count": len(results), "results": results})


# ══════════════════════════════════════════════
#  K-REACH 단독 검색
# ══════════════════════════════════════════════
@app.route("/api/chem", methods=["GET"])
def chem_search():
    if not _req:
        return jsonify({"error": "requests 모듈 없음"}), 500
    search_nm = request.args.get("searchNm", "")
    if not search_nm:
        return jsonify({"error": "searchNm 필요"}), 400
    params = {
        "serviceKey":  KREACH_KEY,
        "searchGubun": request.args.get("searchGubun", "2"),
        "searchNm":    search_nm,
        "pageNo":      request.args.get("pageNo", "1"),
        "numOfRows":   request.args.get("numOfRows", "20"),
        "returnType":  "JSON",
    }
    try:
        r = _req.get(KREACH_URL, params=params, timeout=10)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# ══════════════════════════════════════════════
#  KOSHA 단독 검색 (진단용)
# ══════════════════════════════════════════════
@app.route("/api/kosha/search", methods=["GET"])
def kosha_search_route():
    cas = request.args.get("cas", "").strip()
    if not cas:
        return jsonify({"error": "cas 파라미터 필요"}), 400
    if not kosha_search:
        return jsonify({"error": f"api_kosha 로드 실패: {_import_errors.get('api_kosha','')}"}), 500
    try:
        result = kosha_search(cas)
        return jsonify({"ok": True, "cas": cas, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# ══════════════════════════════════════════════
#  실행 (Railway는 PORT 환경변수 자동 지정)
# ══════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("  MSDS AI OCR 분석 서버 (버전 3)")
    print(f"  http://0.0.0.0:{port}")
    if _import_errors:
        print("  ⚠️  임포트 에러:")
        for mod, err in _import_errors.items():
            print(f"      {mod}: {err}")
    print("  POST /api/msds/analyze  ← MSDS OCR 분석")
    print("  GET  /api/chem          ← K-REACH 검색")
    print("  GET  /api/kosha/search  ← KOSHA 검색")
    print("  GET  /api/debug         ← 진단 정보")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=port)
