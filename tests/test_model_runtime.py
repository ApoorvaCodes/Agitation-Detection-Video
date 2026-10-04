from person1.perception import _model_load_error, runtime_diagnostics

def test_runtime_diagnostics_is_explicit():
    details=runtime_diagnostics("yolo11n.pt","bytetrack")
    assert details["python_executable"] and details["python_version"]
    assert details["model"]=="yolo11n.pt" and details["tracker"]=="bytetrack"

def test_missing_ultralytics_error_is_actionable():
    error=ModuleNotFoundError("No module named 'ultralytics'")
    error.name="ultralytics"
    message=str(_model_load_error("yolo11n.pt","bytetrack",error))
    assert "same Python interpreter" in message
    assert "python_executable=" in message and "ultralytics_importable=" in message
