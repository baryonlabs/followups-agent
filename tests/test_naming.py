from followups import naming


def test_strip_call_default(monkeypatch):
    monkeypatch.delenv("FOLLOWUPS_APP_NAME", raising=False)
    monkeypatch.delenv("FOLLOWUPS_APP_ALIASES", raising=False)
    assert naming.strip_call("@우리공 마포구 점포 수") == ("마포구 점포 수", True)
    assert naming.strip_call("@공, 구별로도") == ("구별로도", True)
    assert naming.strip_call("@공아 고마워") == ("고마워", True)
    assert naming.strip_call("@공") == ("", True)
    assert naming.strip_call("@공장 매출") == ("@공장 매출", False)
    assert naming.strip_call("공 차기") == ("공 차기", False)
    assert naming.call_names() == ["@우리공", "@공"]


def test_rename(monkeypatch):
    monkeypatch.setenv("FOLLOWUPS_APP_NAME", "도우미")
    monkeypatch.setenv("FOLLOWUPS_APP_ALIASES", "도")
    assert naming.call_names() == ["@도우미", "@도"]
    assert naming.strip_call("@도 안녕") == ("안녕", True)
    assert naming.strip_call("@공 안녕")[1] is False
    assert ">도<" in naming.icon_svg()
