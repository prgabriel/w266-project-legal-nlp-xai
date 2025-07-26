import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AzureConfig:
    """Azure-specific configuration"""
    
    # Storage
    storage_account: Optional[str] = os.getenv('AZURE_STORAGE_ACCOUNT')
    storage_key: Optional[str] = os.getenv('AZURE_STORAGE_KEY')
    model_container: str = 'models'
    
    # Application Insights
    instrumentation_key: Optional[str] = os.getenv('APPINSIGHTS_INSTRUMENTATION_KEY')
    
    # Resource limits for Azure Container Apps
    cpu_limit: float = 1.0
    memory_limit: str = '2Gi'
    
    # Scaling
    min_replicas: int = 1
    max_replicas: int = 3
    
    @property
    def is_azure_environment(self) -> bool:
        """Check if running in Azure"""
        return bool(self.storage_account and self.storage_key)

# Global Azure config
azure_config = AzureConfig()