"""main.py 의 /preprocess_intelligent_url(Gena 드라이브 적재용) 과 Gena 설정 로더 단위 테스트.

/preprocess_attachment_url 과 같은 presigned 공용 흐름을 타되, 프로세서만 intelligent_gena_processor
(resource/intelligent_gena_processor_config.yaml)여야 한다. 무거운 facade 초기화는 stub 으로 대체한다.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient


_REPO_ROOT = Path(__file__).resolve().parents[4]
_MAIN_PATH = _REPO_ROOT / "main.py"
_GENA_YAML = _REPO_ROOT / "genon" / "preprocessor" / "resource" / "intelligent_gena_processor_config.yaml"


class _DummyProcessor:
    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def main_module(monkeypatch):
    """무거운 facade 초기화를 대체하고 루트 main.py만 격리 로드한다."""

    class _DummyLogger:
        @staticmethod
        def getLogger(name):
            return logging.getLogger(name)

    class _DummyGenosServiceException(Exception):
        error_code = "1"
        error_msg = "error"

    # GENA_* 가 남아 있으면 임시 사본 생성 경로를 타므로 테스트에서는 제거한다.
    for name in ("GENA_LAYOUT_ENDPOINT", "GENA_LAYOUT_API_KEY", "GENA_OCR_ENDPOINT", "GENA_OCR_MODE"):
        monkeypatch.delenv(name, raising=False)

    stubs = {
        "logger": _module("logger", Logger=_DummyLogger),
        "utils": _module("utils", make_success_response=lambda data=None: {"code": 0, "data": data}),
        "config": _module("config", cors_config=lambda app: None),
        "common.exception": _module(
            "common.exception",
            GenosServiceException=_DummyGenosServiceException,
        ),
        "common.settings": _module(
            "common.settings",
            settings=SimpleNamespace(PREPROCESSOR_ID=None),
        ),
        "util.minio_resource": _module(
            "util.minio_resource",
            download_resource_files=lambda **kwargs: None,
        ),
    }
    for facade_name in (
        "attachment_processor",
        "intelligent_processor",
        "convert_processor",
        "parser_processor",
        "chunking_processor",
    ):
        qualified_name = f"genon.preprocessor.facade.{facade_name}"
        stubs[qualified_name] = _module(qualified_name, DocumentProcessor=_DummyProcessor)

    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_test_root_main_intelligent_url"
    spec = importlib.util.spec_from_file_location(module_name, _MAIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_gena_processor_is_a_separate_instance_loading_gena_yaml(main_module):
    gena = main_module.intelligent_gena_processor
    assert gena is not main_module.intelligent_processor
    assert Path(gena.init_kwargs["config_path"]).name == "intelligent_gena_processor_config.yaml"
    assert Path(gena.init_kwargs["config_path"]) == _GENA_YAML


@pytest.mark.unit
def test_intelligent_url_runs_gena_processor_with_temp_file_and_cleans_up(main_module, monkeypatch):
    captured = {}

    async def fake_download(presigned_url, destination):
        captured["presigned_url"] = presigned_url
        Path(destination).write_bytes(b"%PDF-downloaded-content")
        return len(b"%PDF-downloaded-content")

    async def fake_run(tag, processor, request, file_path, params, marker=None):
        path = Path(file_path)
        captured.update(
            tag=tag,
            processor=processor,
            path=path,
            parent=path.parent,
            content=path.read_bytes(),
            params=params,
        )
        return {"code": 0, "data": [{"text": "intelligent chunk"}]}

    monkeypatch.setattr(main_module, "_download_presigned_file", fake_download)
    monkeypatch.setattr(main_module, "_run", fake_run)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/preprocess_intelligent_url",
            json={
                "presigned_url": "https://storage.example.com/signed?secret=hidden",
                "file_name": "../deck.pptx",
                "params": {"chunk_size": 1000, "chunk_overlap": 100},
            },
        )

    assert response.status_code == 200
    assert response.json() == {"code": 0, "data": [{"text": "intelligent chunk"}]}
    assert captured["presigned_url"] == "https://storage.example.com/signed?secret=hidden"
    assert captured["tag"] == "preprocess_intelligent_url"
    assert captured["processor"] is main_module.intelligent_gena_processor
    assert captured["processor"] is not main_module.attachment_processor
    assert captured["path"].name == "deck.pptx"
    assert captured["parent"].name.startswith("intelligent_url_")
    assert captured["content"] == b"%PDF-downloaded-content"
    # params 는 그대로 전달된다(file_name 등 주입 없음 — Gena 계약 유지).
    assert captured["params"] == {"chunk_size": 1000, "chunk_overlap": 100}
    assert not captured["parent"].exists()


@pytest.mark.unit
def test_intelligent_url_rejects_file_name_without_extension(main_module, monkeypatch):
    async def unexpected_download(*args, **kwargs):
        raise AssertionError("download must not run for invalid file_name")

    monkeypatch.setattr(main_module, "_download_presigned_file", unexpected_download)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/preprocess_intelligent_url",
            json={"presigned_url": "https://storage.example.com/x", "file_name": "noext"},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["code"] == 1
    assert body["error_code"] == main_module.ERROR_CODE_INPUT
    assert body["tag"] == "preprocess_intelligent_url"


@pytest.mark.unit
def test_attachment_url_still_uses_attachment_processor(main_module, monkeypatch):
    """공용 헬퍼로 리팩터링한 뒤에도 첨부 엔드포인트의 프로세서/태그 계약은 그대로다."""
    captured = {}

    async def fake_download(presigned_url, destination):
        Path(destination).write_bytes(b"x")
        return 1

    async def fake_run(tag, processor, request, file_path, params, marker=None):
        captured.update(tag=tag, processor=processor, parent=Path(file_path).parent)
        return {"code": 0, "data": []}

    monkeypatch.setattr(main_module, "_download_presigned_file", fake_download)
    monkeypatch.setattr(main_module, "_run", fake_run)

    with TestClient(main_module.app) as client:
        client.post(
            "/preprocess_attachment_url",
            json={"presigned_url": "https://storage.example.com/x", "file_name": "a.pdf"},
        )

    assert captured["tag"] == "preprocess_attachment_url"
    assert captured["processor"] is main_module.attachment_processor
    assert captured["parent"].name.startswith("attachment_url_")


# ── Gena 설정 로더 ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_materialize_gena_config_returns_source_when_no_env(main_module, tmp_path):
    source = tmp_path / "intelligent_gena_processor_config.yaml"
    source.write_text("layout: {genos_layout: {endpoint: 'http://dev'}}\n", encoding="utf-8")
    assert main_module._materialize_gena_config(str(source), env={}) == str(source)
    # 빈 값은 미설정으로 본다.
    assert main_module._materialize_gena_config(
        str(source), env={"GENA_LAYOUT_ENDPOINT": "  "}
    ) == str(source)


@pytest.mark.unit
def test_materialize_gena_config_applies_env_overrides_and_absolutizes_prompt_files(main_module, tmp_path):
    (tmp_path / "prompt_toc_default_system.md").write_text("sys", encoding="utf-8")
    source = tmp_path / "intelligent_gena_processor_config.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "ocr": {"ocr_mode": "disable", "paddle": {"ocr_endpoint": "http://dev-paddle"}},
                "layout": {"genos_layout": {"endpoint": "http://dev-dots", "api_key": ""}},
                "enrichment": [
                    {"toc": {"enable": False, "system_prompt_file": "prompt_toc_default_system.md",
                             "user_prompt_file": "missing_prompt.md"}},
                ],
                "chunking": {"tokenizer_path": "/models/tok"},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    env = {
        "GENA_LAYOUT_ENDPOINT": "http://prod-dots/v1/chat/completions",
        "GENA_LAYOUT_API_KEY": "k",
        "GENA_OCR_MODE": "auto",
    }

    out_path = main_module._materialize_gena_config(str(source), env=env)

    assert out_path != str(source)
    assert Path(out_path).name == "intelligent_gena_processor_config.yaml"
    cfg = yaml.safe_load(Path(out_path).read_text(encoding="utf-8"))
    assert cfg["layout"]["genos_layout"]["endpoint"] == "http://prod-dots/v1/chat/completions"
    assert cfg["layout"]["genos_layout"]["api_key"] == "k"
    assert cfg["ocr"]["ocr_mode"] == "auto"
    assert cfg["ocr"]["paddle"]["ocr_endpoint"] == "http://dev-paddle"  # 미지정 키는 유지
    toc = cfg["enrichment"][0]["toc"]
    # 존재하는 프롬프트 파일은 원본 디렉터리 기준 절대 경로로, 없는 파일은 그대로 둔다.
    assert toc["system_prompt_file"] == str((tmp_path / "prompt_toc_default_system.md").resolve())
    assert toc["user_prompt_file"] == "missing_prompt.md"
    assert cfg["chunking"]["tokenizer_path"] == "/models/tok"  # *_file 이 아닌 키는 건드리지 않음
    # 원본은 변경되지 않는다.
    original = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert original["layout"]["genos_layout"]["endpoint"] == "http://dev-dots"


# ── Gena yaml 1단계 값 가드 ───────────────────────────────────────────────────

@pytest.mark.unit
def test_gena_yaml_stage1_values():
    cfg = yaml.safe_load(_GENA_YAML.read_text(encoding="utf-8"))

    assert cfg["ocr"]["ocr_mode"] == "disable"
    assert cfg["layout"]["layout_model_type"] == "genos_layout"
    assert cfg["layout"]["genos_layout"]["endpoint"].startswith("http")
    assert cfg["pdf_pipeline"]["generate_picture_images"] is False
    assert cfg["formats"]["ppt"]["page_description"]["enable"] is False
    assert cfg["table_image"]["enable"] is False
    assert cfg["chunking"]["chunk_mode"] == "split_only"

    enrichment = cfg["enrichment"]
    assert isinstance(enrichment, list) and enrichment
    for entry in enrichment:
        (name, options), = entry.items()
        assert options.get("enable") is False, f"enrichment.{name} 은 1단계에서 off 여야 한다"
        chart = options.get("chart")
        if isinstance(chart, dict):
            assert chart.get("enable") is False
        refine = options.get("refine")
        if isinstance(refine, dict):
            assert refine.get("enable") is False

    # 참조하는 프롬프트 파일은 같은 resource/ 디렉터리에 실제로 있어야 한다(비활성이라도 로더가 읽는다).
    resource_dir = _GENA_YAML.parent

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and str(key).endswith("_file"):
                    assert (resource_dir / value).is_file(), f"{key}={value} 가 resource/ 에 없다"
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(cfg)
