"""attachment_processor 의 문자수 분할 헬퍼 `_char_split_text` 단위 테스트.

전에는 이 파일이 두 가지를 더 했는데 둘 다 걷어냈다.

  - 소스에서 AST 로 함수만 뽑아 exec 하는 로더를 썼다. "로컬은 vendored docling 충돌로
    facade.attachment_processor import 가 불가하다"는 전제였는데 지금은 정상 import 된다
    (같은 디렉터리의 test_attachment_chunk_config_unit.py 가 이미 importorskip 으로 쓴다).
    전제가 사라졌으므로 평범한 import 로 되돌렸다.
  - 출고 yaml 의 키 배치를 단정하는 TestConfigStructure 가 있었다. 끝난 이관(chunk_size
    공통화, generic·token cap 제거, hwp 옵션의 formats 이동)이 유지되는지 보는 체크리스트였다.
    같은 내용을 test_attachment_chunk_config_unit.py 가 **동작으로** 검증하고, 그쪽은 출고
    config 를 복사해 쓰므로 키가 사라지면 거기서 먼저 깨진다. 구조 단정은 중복인 데다
    `set(defaults) <= {...}` 같은 형태라 새 설정 키를 못 늘리게 막고 있었다.

의존성(docling 등) 미가용 환경에서는 importorskip 으로 자동 skip 된다(CI gate).
"""

import pytest

attachment = pytest.importorskip("facade.attachment_processor")

_char_split_text = attachment._char_split_text

_TXT = "abcdefghij" * 5  # 50자


@pytest.mark.unit
class TestCharSplitText:
    def test_empty_returns_empty(self):
        assert _char_split_text("", chunk_size=0, chunk_overlap=0) == []

    def test_chunk_size_zero_single_chunk(self):
        """chunk_size=0 → 문서 전체가 1청크(분할 안 함)."""
        assert _char_split_text(_TXT, chunk_size=0, chunk_overlap=100) == [_TXT]

    def test_chunk_size_none_single_chunk(self):
        """chunk_size 미지정(None) → 0 과 동일하게 1청크."""
        assert _char_split_text(_TXT, chunk_size=None, chunk_overlap=None) == [_TXT]

    def test_positive_chunk_size_splits_by_chars(self):
        """chunk_size>0 → 문자수 단위 분할, 각 청크는 chunk_size 이하, 원문 보존."""
        out = _char_split_text(_TXT, chunk_size=20, chunk_overlap=0)
        assert len(out) > 1
        assert all(len(c) <= 20 for c in out)
        assert "".join(out) == _TXT
