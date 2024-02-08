import unittest
import warnings
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from src.app import app

# ignore warnings
warnings.filterwarnings("ignore")

bad_response_200 = "response should be 200"
bad_response_400 = "response should be 400"


class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home(self):
        response = self.app.get("/", follow_redirects=True)
        self.assertEqual(response.status_code, 200, bad_response_200)

    def test_monitoring(self):
        response = self.app.get("/monitoring", follow_redirects=True)
        self.assertEqual(response.status_code, 200, bad_response_200)

    def test_predict(self):
        response = self.app.post(
            "/predict", data=dict(post="test"), follow_redirects=True
        )
        self.assertEqual(response.status_code, 200, bad_response_200)

    def test_predict_missing_post(self):
        response = self.app.post("/predict", follow_redirects=True)
        self.assertEqual(response.status_code, 400, bad_response_400)

    def test_upload_csv_for_monitoring(self):
        response = self.app.post(
            "/monitoring",
            data={"file": (open("tests/unit/test_data.csv", "rb"), "test_data.csv")},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200, bad_response_200)

    def test_upload_csv_for_monitoring_missing_file(self):
        response = self.app.post("/monitoring", follow_redirects=True)
        print(response)
        self.assertEqual(response.status_code, 200, bad_response_200)

    def test_upload_csv_for_monitoring_invalid_file(self):
        response = self.app.post(
            "/monitoring",
            data={"file": (open("tests/unit/test_data.txt", "rb"), "test_data.txt")},
            follow_redirects=True,
        )
        self.assertEqual(
            response.data, b"Invalid file type, please upload a CSV.", bad_response_400
        )


if __name__ == "__main__":
    unittest.main()
