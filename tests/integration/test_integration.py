from selenium.webdriver.common.by import By
from selenium import webdriver

import unittest
import os


class TestIntegration(unittest.TestCase):
    def setUp(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(options)
        self.driver.get("http://localhost:5000")

    def teardown_method(self, method):
        self.driver.quit()

    def test_predict(self):
        self.driver.get("http://127.0.0.1:5000/")
        self.driver.find_element(By.ID, "post").click()
        self.driver.find_element(By.ID, "post").send_keys("test")
        self.driver.find_element(By.ID, "submit").click()

    def test_monitoring(self):
        self.driver.get("http://127.0.0.1:5000/monitoring")
        self.driver.find_element(By.ID, "file").send_keys(
            os.path.abspath("tests/test_data.csv")
        )
        self.driver.find_element(By.ID, "submit").click()


if __name__ == "__main__":
    unittest.main()
