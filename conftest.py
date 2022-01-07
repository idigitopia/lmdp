from pytest import fixture

# content of conftest.py
import pytest
import smtplib

@pytest.fixture
def input_value():
   input = 39
   return input

# @pytest.fixture(scope="module", params=["smtp.gmail.com", "mail.python.org"])
# def smtp_connection(request):
#     smtp_connection = smtplib.SMTP(request.param, 587, timeout=5)
#     yield smtp_connection
#     print("finalizing {}".format(smtp_connection))
#     smtp_connection.close()



@pytest.fixture
def standard_buffer_instance():
    """
    get empty buffer of default state shape and action shape.
    """
    from lmdp.data.buffer import StandardBuffer
    import numpy as np 
    
    # instantiate StandardBuffer 
    state_shape = [2,2]
    action_shape = [4]
    buffer_size = 1000
    device =  "cpu"
    standard_buffer = StandardBuffer(state_shape = state_shape,
                          action_shape = action_shape,
                          buffer_size = buffer_size,
                          device = device)

    return standard_buffer
