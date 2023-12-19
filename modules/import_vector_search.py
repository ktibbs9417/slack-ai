import os
import urllib.request

class ImportVectorSearch:

    def import_vertex_search(self):
        if os.path.exists("utils"):
            print("utils directory already exists.")
        elif not os.path.exists("utils"):
            print("utils directory does not exist. Creating utils directory.")
            os.makedirs("utils")
            url_prefix = "https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/use-cases/document-qa/utils"
            files = ["__init__.py", "matching_engine.py", "matching_engine_utils.py"]

            for fname in files:
                urllib.request.urlretrieve(f"{url_prefix}/{fname}", filename=f"utils/{fname}")