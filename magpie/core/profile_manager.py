"""
Profile management for Magpie.

Handles loading and saving user profiles to/from JSON files.
"""

import json
import os
import typing

from magpie.models.profile import UserProfile
from magpie.utils.config import Config


def load_profile(user_id: typing.Optional[str] = None) -> UserProfile:
    """
    Load user profile from disk, or create new profile if doesn't exist.
    
    Args:
        user_id: User identifier. If None, uses Config.DEFAULT_USER_ID
        
    Returns:
        UserProfile loaded from file or newly created
        
    Example:
        >>> profile = load_profile("user_123")
        >>> profile = load_profile()  # Uses default user
    """
    if user_id is None:
        user_id = Config.DEFAULT_USER_ID
    
    profile_path = _get_profile_path(user_id)
    
    # If profile exists, load it
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r') as f:
                json_data = f.read()
                profile = UserProfile.model_validate_json(json_data)
                return profile
        except Exception as e:
            print(f"Warning: Failed to load profile from {profile_path}: {e}")
            print("Creating new profile instead.")
    
    # Otherwise create new profile
    return create_profile(user_id)


def save_profile(profile: UserProfile) -> None:
    """
    Save user profile to disk as JSON.
    
    Creates profile directory if it doesn't exist.
    
    Args:
        profile: UserProfile to save
        
    Example:
        >>> profile = UserProfile(user_id="user_123")
        >>> save_profile(profile)
    """
    # Ensure profile directory exists
    profile_dir = Config.PROFILE_DIR
    os.makedirs(profile_dir, exist_ok=True)
    
    profile_path = _get_profile_path(profile.user_id)
    
    # Serialize to JSON
    json_data = profile.model_dump_json(indent=2)
    
    # Write to file
    with open(profile_path, 'w') as f:
        f.write(json_data)


def create_profile(user_id: str) -> UserProfile:
    """
    Create a new user profile with default settings.
    
    Does not save to disk - call save_profile() to persist.
    
    Args:
        user_id: Unique identifier for user
        
    Returns:
        New UserProfile with defaults
        
    Example:
        >>> profile = create_profile("user_123")
        >>> save_profile(profile)
    """
    profile = UserProfile(user_id=user_id)
    return profile


def delete_profile(user_id: str) -> bool:
    """
    Delete user profile from disk.
    
    Args:
        user_id: User identifier
        
    Returns:
        True if deleted, False if profile didn't exist
        
    Example:
        >>> delete_profile("user_123")
        True
    """
    profile_path = _get_profile_path(user_id)
    
    if os.path.exists(profile_path):
        os.remove(profile_path)
        return True
    
    return False


def profile_exists(user_id: str) -> bool:
    """
    Check if a profile exists on disk.
    
    Args:
        user_id: User identifier
        
    Returns:
        True if profile file exists, False otherwise
    """
    profile_path = _get_profile_path(user_id)
    return os.path.exists(profile_path)


def list_profiles() -> typing.List[str]:
    """
    List all user IDs that have profiles.
    
    Returns:
        List of user_id strings
        
    Example:
        >>> list_profiles()
        ['default_user', 'user_123', 'user_456']
    """
    profile_dir = Config.PROFILE_DIR
    
    if not os.path.exists(profile_dir):
        return []
    
    profiles = []
    for filename in os.listdir(profile_dir):
        if filename.endswith('.json'):
            user_id = filename[:-5]  # Remove .json extension
            profiles.append(user_id)
    
    return profiles


def _get_profile_path(user_id: str) -> str:
    """
    Get file path for a user's profile.
    
    Args:
        user_id: User identifier
        
    Returns:
        Full path to profile JSON file
    """
    profile_dir = Config.PROFILE_DIR
    filename = f"{user_id}.json"
    return os.path.join(profile_dir, filename)
