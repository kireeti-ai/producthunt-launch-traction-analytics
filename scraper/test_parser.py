import os
import sys
from scraper.parser import parse_product_page

def test_parser():
    print("=== Testing Product Hunt HTML Parser ===")
    sample_file = "wisprflow_sample.html"
    if not os.path.exists(sample_file):
        print(f"Error: Cache file {sample_file} not found. Please run fetch scratch scripts first.")
        sys.exit(1)
        
    with open(sample_file, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    url = "https://www.producthunt.com/products/wisprflow"
    tagline = "Speak naturally, write perfectly & 4x faster in every app"
    
    # Run the parser
    product = parse_product_page(html_content, url)
    product.compute_derived(tagline)
    
    # Assertions and reports
    print(f"URL: {product.url}")
    print(f"Description: {product.description}")
    print(f"Launch Timestamp: {product.launch_timestamp}")
    print(f"Launch Hour: {product.launch_hour}")
    print(f"Hunter: {product.hunter_name} (Expected None due to PH layout redirect)")
    print(f"Makers: {product.maker_names}")
    print(f"Maker Profile URLs: {product.maker_profile_urls}")
    print(f"Team Size: {product.team_size}")
    print(f"Media Count: {product.media_count}")
    print(f"Website URL: {product.website_url}")
    print(f"Domain: {product.product_domain}")
    print(f"Is Featured: {product.is_featured}")
    print(f"Self Launched: {product.self_launched}")
    print(f"Description Length: {product.description_length}")
    print(f"Tagline Length: {product.tagline_length}")
    
    # Validate fields
    assert product.description is not None, "Description should not be None"
    assert "voice" in product.description.lower(), "Description should contain details about voice"
    assert product.launch_timestamp is not None, "Launch timestamp should not be None"
    assert product.launch_hour is not None, "Launch hour should not be None"
    assert "Tanay Kothari" in product.maker_names, "Tanay Kothari should be in maker names"
    assert len(product.maker_profile_urls) == len(product.maker_names), "Profile URLs length should match maker names count"
    assert product.media_count == 3, "Media count should be 3"
    assert product.website_url == "https://wisprflow.ai", "Website URL should be https://wisprflow.ai"
    assert product.product_domain == "wisprflow.ai", "Domain should be wisprflow.ai"
    assert product.is_featured is True, "Is Featured should be True"
    assert product.description_length > 100, "Description length should be greater than 100"
    assert product.tagline_length > 10, "Tagline length should be greater than 10"
    
    print("\nAll unit tests passed successfully!")

if __name__ == "__main__":
    test_parser()
