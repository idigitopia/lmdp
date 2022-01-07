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
