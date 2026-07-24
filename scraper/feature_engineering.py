import json
import math
from datetime import timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import pandas as pd


def safe_json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def safe_json_loads(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def extract_product_slug(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    parts = [part for part in urlparse(value).path.split('/') if part]
    if 'products' not in parts:
        return None
    index = parts.index('products')
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def parse_utc_timestamp(value: Any) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pd.NaT
    parsed = pd.to_datetime(value, errors='coerce', utc=True)
    if not pd.isna(parsed):
        return parsed
    try:
        from dateutil import parser as dateutil_parser

        return pd.Timestamp(dateutil_parser.isoparse(str(value))).tz_convert(timezone.utc)
    except Exception:
        return pd.NaT


def website_domain_from_url(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    parsed = urlparse(value)
    domain = parsed.netloc.lower()
    if not domain:
        return None
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain


def _as_list(value: Any) -> list:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        loaded = safe_json_loads(value)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            return [loaded]
    return []


def flatten_topic_edges(topic_edges: Any) -> Dict[str, Any]:
    edges = _as_list(topic_edges)
    topics = []
    for edge in edges:
        node = edge.get('node') if isinstance(edge, dict) else None
        if not isinstance(node, dict):
            continue
        topics.append(
            {
                'id': node.get('id'),
                'slug': node.get('slug'),
                'name': node.get('name'),
                'description': node.get('description'),
                'followersCount': node.get('followersCount'),
                'postsCount': node.get('postsCount'),
                'url': node.get('url'),
                'isFollowing': node.get('isFollowing'),
                'createdAt': node.get('createdAt'),
            }
        )

    return {
        'topic_count': len(topics),
        'topic_ids': safe_json_dumps([topic.get('id') for topic in topics]),
        'topic_slugs': safe_json_dumps([topic.get('slug') for topic in topics]),
        'topic_names': safe_json_dumps([topic.get('name') for topic in topics]),
        'topic_objects': safe_json_dumps(topics),
        'primary_topic': topics[0].get('name') if topics else None,
    }


def flatten_user(user: Any, prefix: str) -> Dict[str, Any]:
    if not isinstance(user, dict):
        return {
            f'{prefix}_id': None,
            f'{prefix}_username': None,
            f'{prefix}_name': None,
            f'{prefix}_headline': None,
            f'{prefix}_followers_count': None,
            f'{prefix}_following_count': None,
            f'{prefix}_website_url': None,
            f'{prefix}_twitter_username': None,
            f'{prefix}_profile_url': None,
            f'{prefix}_is_maker': None,
            f'{prefix}_is_following': None,
        }

    return {
        f'{prefix}_id': user.get('id'),
        f'{prefix}_username': user.get('username'),
        f'{prefix}_name': user.get('name'),
        f'{prefix}_headline': user.get('headline'),
        f'{prefix}_followers_count': user.get('followersCount'),
        f'{prefix}_following_count': user.get('followingCount'),
        f'{prefix}_website_url': user.get('websiteUrl'),
        f'{prefix}_twitter_username': user.get('twitterUsername'),
        f'{prefix}_profile_url': user.get('url'),
        f'{prefix}_is_maker': user.get('isMaker'),
        f'{prefix}_is_following': user.get('isFollowing'),
    }


def flatten_makers(makers: Any) -> Dict[str, Any]:
    maker_items = [maker for maker in _as_list(makers) if isinstance(maker, dict)]
    maker_payload = []
    for maker in maker_items:
        maker_payload.append(
            {
                'id': maker.get('id'),
                'username': maker.get('username'),
                'name': maker.get('name'),
                'headline': maker.get('headline'),
                'followersCount': maker.get('followersCount'),
                'followingCount': maker.get('followingCount'),
                'websiteUrl': maker.get('websiteUrl'),
                'twitterUsername': maker.get('twitterUsername'),
                'url': maker.get('url'),
                'isMaker': maker.get('isMaker'),
                'isFollowing': maker.get('isFollowing'),
            }
        )

    return {
        'maker_count': len(maker_payload),
        'maker_ids': safe_json_dumps([maker.get('id') for maker in maker_payload]),
        'maker_usernames': safe_json_dumps([maker.get('username') for maker in maker_payload]),
        'maker_names_json': safe_json_dumps([maker.get('name') for maker in maker_payload]),
        'maker_headlines': safe_json_dumps([maker.get('headline') for maker in maker_payload]),
        'maker_followers_counts': safe_json_dumps([maker.get('followersCount') for maker in maker_payload]),
        'maker_total_followers': sum(int(maker.get('followersCount') or 0) for maker in maker_payload),
        'maker_total_following': sum(int(maker.get('followingCount') or 0) for maker in maker_payload),
        'maker_profile_urls': safe_json_dumps([maker.get('url') for maker in maker_payload]),
        'maker_twitter_usernames': safe_json_dumps([maker.get('twitterUsername') for maker in maker_payload]),
        'maker_objects': safe_json_dumps(maker_payload),
    }


def flatten_media(media: Any) -> Dict[str, Any]:
    media_items = [item for item in _as_list(media) if isinstance(item, dict)]
    image_count = sum(1 for item in media_items if item.get('type') == 'image')
    video_count = sum(1 for item in media_items if item.get('type') == 'video')
    return {
        'media_count': len(media_items),
        'image_count': image_count,
        'video_count': video_count,
        'has_video': video_count > 0,
        'screenshots_count': image_count,
        'media_objects': safe_json_dumps(media_items),
    }


def flatten_thumbnail(thumbnail: Any) -> Dict[str, Any]:
    if not isinstance(thumbnail, dict):
        return {
            'thumbnail_type': None,
            'thumbnail_url': None,
            'thumbnail_video_url': None,
        }
    return {
        'thumbnail_type': thumbnail.get('type'),
        'thumbnail_url': thumbnail.get('url'),
        'thumbnail_video_url': thumbnail.get('videoUrl'),
    }


def flatten_product_links(product_links: Any) -> Dict[str, Any]:
    links = [link for link in _as_list(product_links) if isinstance(link, dict)]
    website_link = next((link for link in links if str(link.get('type')).lower() == 'website'), None)
    return {
        'product_links_count': len(links),
        'product_links_json': safe_json_dumps(links),
        'ph_website_url': website_link.get('url') if website_link else None,
    }


def flatten_collections(collections: Any) -> Dict[str, Any]:
    edges = _as_list(collections)
    items = []
    for edge in edges:
        node = edge.get('node') if isinstance(edge, dict) else None
        if not isinstance(node, dict):
            continue
        items.append(
            {
                'id': node.get('id'),
                'name': node.get('name'),
                'description': node.get('description'),
                'featuredAt': node.get('featuredAt'),
                'followersCount': node.get('followersCount'),
                'url': node.get('url'),
                'user': node.get('user'),
            }
        )
    return {
        'collection_count': len(items),
        'collection_names': safe_json_dumps([item.get('name') for item in items]),
        'collection_objects': safe_json_dumps(items),
    }


def normalize_launch_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    if 'url' in frame.columns and 'product_slug' not in frame.columns:
        frame['product_slug'] = frame['url'].map(extract_product_slug)

    if 'launch_timestamp' in frame.columns:
        frame['launch_timestamp_utc'] = frame['launch_timestamp'].map(parse_utc_timestamp)
    elif 'createdAt' in frame.columns:
        frame['launch_timestamp_utc'] = frame['createdAt'].map(parse_utc_timestamp)
    else:
        frame['launch_timestamp_utc'] = pd.NaT

    frame['launch_date'] = frame['launch_timestamp_utc'].dt.date
    frame['launch_hour'] = frame['launch_timestamp_utc'].dt.hour
    frame['weekday'] = frame['launch_timestamp_utc'].dt.dayofweek
    frame['month'] = frame['launch_timestamp_utc'].dt.month
    frame['week_of_year'] = frame['launch_timestamp_utc'].dt.isocalendar().week.astype('Int64')
    frame['weekend'] = frame['weekday'].isin([5, 6])

    if 'description' in frame.columns:
        frame['description_length'] = frame['description'].fillna('').astype(str).str.len()
    if 'tagline' in frame.columns:
        frame['tagline_length'] = frame['tagline'].fillna('').astype(str).str.len()

    if 'topic_count' in frame.columns:
        frame['topic_count'] = pd.to_numeric(frame['topic_count'], errors='coerce').fillna(0).astype(int)
    if 'maker_count' in frame.columns:
        frame['maker_count'] = pd.to_numeric(frame['maker_count'], errors='coerce').fillna(0).astype(int)
    if 'maker_total_followers' in frame.columns:
        frame['maker_total_followers'] = pd.to_numeric(frame['maker_total_followers'], errors='coerce').fillna(0).astype(int)
    if 'media_count' in frame.columns:
        frame['media_count'] = pd.to_numeric(frame['media_count'], errors='coerce').fillna(0).astype(int)

    if 'website_url' in frame.columns:
        frame['website_domain'] = frame['website_url'].map(website_domain_from_url)
    elif 'ph_website_url' in frame.columns:
        frame['website_domain'] = frame['ph_website_url'].map(website_domain_from_url)
    else:
        frame['website_domain'] = None

    if 'featuredAt' in frame.columns:
        frame['is_featured'] = frame['featuredAt'].notna()
    elif 'is_featured' in frame.columns:
        frame['is_featured'] = frame['is_featured'].astype(bool)
    else:
        frame['is_featured'] = False

    if 'votesCount' in frame.columns:
        vote_series = pd.to_numeric(frame['votesCount'], errors='coerce')
        frame['high_traction'] = False
        if 'launch_date' in frame.columns:
            for _, index_values in frame.groupby('launch_date').groups.items():
                group_votes = vote_series.loc[index_values]
                if group_votes.notna().any():
                    threshold = group_votes.quantile(0.8)
                    frame.loc[index_values, 'high_traction'] = group_votes >= threshold
    elif 'upvotes' in frame.columns:
        vote_series = pd.to_numeric(frame['upvotes'], errors='coerce')
        frame['high_traction'] = False
        if 'launch_date' in frame.columns:
            for _, index_values in frame.groupby('launch_date').groups.items():
                group_votes = vote_series.loc[index_values]
                if group_votes.notna().any():
                    threshold = group_votes.quantile(0.8)
                    frame.loc[index_values, 'high_traction'] = group_votes >= threshold
    else:
        frame['high_traction'] = False

    return frame
