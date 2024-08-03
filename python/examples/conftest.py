import pytest
import os
import tempfile
import triton
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

def empty_decorator(func):
    return func

pytest.mark.interpreter = empty_decorator

@pytest.fixture
def device(request):
    return "cpu"