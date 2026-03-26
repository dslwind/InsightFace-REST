import unittest

from pydantic import ValidationError

from if_rest.schemas import BodyExtract, FaceVerification, Images


class ImagesSchemaTests(unittest.TestCase):
    def test_images_require_exactly_one_source(self):
        with self.assertRaises(ValidationError):
            Images()

        with self.assertRaises(ValidationError):
            Images(data=["abc"], urls=["img.jpg"])

    def test_images_accept_single_source(self):
        from_data = Images(data=["abc"])
        self.assertEqual(from_data.data, ["abc"])
        self.assertIsNone(from_data.urls)

        from_urls = Images(urls=["img.jpg"])
        self.assertEqual(from_urls.urls, ["img.jpg"])
        self.assertIsNone(from_urls.data)


class FaceVerificationSchemaTests(unittest.TestCase):
    def test_face_verification_requires_two_images(self):
        with self.assertRaises(ValidationError):
            FaceVerification(images=Images(data=["abc"]))

    def test_face_verification_accepts_two_images(self):
        schema = FaceVerification(images=Images(data=["abc", "def"]))
        self.assertEqual(schema.det_threshold, 0.6)


class BodyExtractSchemaTests(unittest.TestCase):
    def test_img_headers_default_is_isolated(self):
        first = BodyExtract(images=Images(data=["abc"]))
        second = BodyExtract(images=Images(data=["def"]))

        first.img_req_headers["Authorization"] = "token"
        self.assertEqual(second.img_req_headers, {})


if __name__ == "__main__":
    unittest.main()
