from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

@dataclass
class EnrichedProduct:
    url: str
    description: Optional[str] = None
    launch_timestamp: Optional[str] = None
    launch_hour: Optional[int] = None
    hunter_name: Optional[str] = None
    hunter_profile_url: Optional[str] = None
    maker_names: List[str] = field(default_factory=list)
    maker_profile_urls: List[str] = field(default_factory=list)
    team_size: int = 0
    media_count: int = 0
    website_url: Optional[str] = None
    product_domain: Optional[str] = None
    is_featured: bool = False
    self_launched: bool = False
    
    # Derived features
    description_length: int = 0
    tagline_length: int = 0

    def compute_derived(self, tagline: Optional[str] = None):
        """
        Computes the derived fields from the base scraped data.
        """
        # Description length
        self.description_length = len(self.description) if self.description else 0
        
        # Tagline length
        self.tagline_length = len(tagline) if tagline else 0
        
        # Team size from parsed makers list
        if self.maker_names:
            self.team_size = len(self.maker_names)
            
        # Self launched check
        if self.hunter_name and self.maker_names:
            # Check case-insensitively
            hunter_lower = self.hunter_name.strip().lower()
            self.self_launched = any(
                m.strip().lower() == hunter_lower 
                for m in self.maker_names
            )
        else:
            self.self_launched = False

        # Domain extraction fallback
        if self.website_url and not self.product_domain:
            try:
                parsed_uri = urlparse(self.website_url)
                domain = parsed_uri.netloc
                if domain.startswith("www."):
                    domain = domain[4:]
                self.product_domain = domain
            except Exception:
                self.product_domain = None

    def to_dict(self) -> dict:
        """
        Converts the data class to a dictionary for pandas integration.
        """
        return {
            "description": self.description,
            "launch_timestamp": self.launch_timestamp,
            "launch_hour": self.launch_hour,
            "hunter_name": self.hunter_name,
            "hunter_profile_url": self.hunter_profile_url,
            "maker_names": ", ".join(self.maker_names) if self.maker_names else None,
            "maker_profile_urls": ", ".join(self.maker_profile_urls) if self.maker_profile_urls else None,
            "team_size": self.team_size,
            "media_count": self.media_count,
            "website_url": self.website_url,
            "product_domain": self.product_domain,
            "is_featured": self.is_featured,
            "self_launched": self.self_launched,
            "description_length": self.description_length,
            "tagline_length": self.tagline_length
        }
