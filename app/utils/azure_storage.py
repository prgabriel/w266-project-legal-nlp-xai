# app/utils/azure_storage.py
from azure.storage.blob import BlobServiceClient
import os

class AzureModelManager:
    def __init__(self):
        self.blob_service = BlobServiceClient(
            account_url=f"https://{azure_config.storage_account}.blob.core.windows.net",
            credential=azure_config.storage_key
        )
    
    def download_model(self, model_name: str, local_path: str):
        """Download model from Azure Blob Storage"""
        blob_client = self.blob_service.get_blob_client(
            container=azure_config.model_container, 
            blob=f"{model_name}.tar.gz"
        )
        
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())