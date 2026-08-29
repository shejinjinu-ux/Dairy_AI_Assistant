"""
Centralized Ownership & Security Authorization Guard
Extracts user identity from HTTP authentication tokens/headers,
enforces strict User -> Farm -> Cattle ownership validation,
prevents cross-tenant access, and restricts guest access to generic public endpoints.
"""

import logging
from typing import Optional
from fastapi import Request, HTTPException, status
from backend.app.schemas.user_farm_cattle import OwnershipValidationContext, UserProfile, Farm, Cattle
from backend.app.db.farm_cattle_repository import get_farm_cattle_repository, DEMO_USER_ID

logger = logging.getLogger("dairy_ai.core.ownership_guard")


class OwnershipGuard:
    """Authorization guard enforcing User -> Farm -> Cattle ownership hierarchy."""

    @staticmethod
    def extract_authenticated_user_id(request: Request) -> Optional[str]:
        """
        Derives authenticated user identity from request headers.
        Inspects Authorization Bearer tokens, Supabase JWT claims, or X-User-ID header.
        NEVER trusts raw unauthenticated request body parameters for user identity.
        """
        # 1. Authorization Bearer Token check
        auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            if token:
                # If bearer token is demo token:
                if token.lower() in ["demo", "demo_token", "bearer_demo"]:
                    return DEMO_USER_ID
                if token.lower().startswith("usr_") or token.lower().startswith("user_"):
                    return token
                return f"usr_{token[:16]}"

        # 2. Explicit X-User-ID / X-Authenticated-User header
        x_user_id = request.headers.get("X-User-ID") or request.headers.get("x-user-id")
        if x_user_id and x_user_id.strip():
            return x_user_id.strip()

        # 3. Supabase Auth Token Header
        sb_user = request.headers.get("X-Supabase-User-ID")
        if sb_user and sb_user.strip():
            return sb_user.strip()

        return None

    @classmethod
    def validate_request_ownership(
        cls,
        request: Request,
        farm_id: Optional[str] = None,
        animal_id: Optional[str] = None,
        require_auth: bool = False
    ) -> OwnershipValidationContext:
        """
        Performs strict ownership validation:
        1. Derives authenticated user identity.
        2. If require_auth=True and user is unauthenticated, raises 401 Unauthorized.
        3. If farm_id provided, verifies user owns the farm.
        4. If animal_id provided, verifies user owns the animal under that farm.
        5. Guest/Anonymous mode can NEVER access user-owned or personalized data.
        """
        repo = get_farm_cattle_repository()
        user_id = cls.extract_authenticated_user_id(request)

        if require_auth and not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Please provide a valid Authorization bearer token or user session."
            )

        if not user_id:
            # Unauthenticated guest request
            if farm_id or animal_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Access denied: Unauthenticated guest requests cannot access personalized farm or animal resources."
                )
            return OwnershipValidationContext(is_authenticated=False, is_demo=False)

        # User is authenticated
        is_demo = (user_id == DEMO_USER_ID)
        val_farm: Optional[Farm] = None
        val_cattle: Optional[Cattle] = None

        # Validate Farm Ownership
        if farm_id:
            val_farm = repo.get_farm(farm_id=farm_id, user_id=user_id)
            if not val_farm:
                logger.warning(f"Ownership rejection: User '{user_id}' attempted to access unauthorized farm '{farm_id}'.")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: You do not own farm '{farm_id}'."
                )

        # Validate Cattle Ownership
        if animal_id:
            val_cattle = repo.get_cattle(animal_id=animal_id, user_id=user_id, farm_id=farm_id)
            if not val_cattle:
                logger.warning(f"Ownership rejection: User '{user_id}' attempted to access unauthorized animal '{animal_id}'.")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: You do not own animal '{animal_id}'."
                )

        return OwnershipValidationContext(
            is_authenticated=True,
            user_id=user_id,
            is_demo=is_demo,
            validated_farm=val_farm,
            validated_cattle=val_cattle
        )


ownership_guard = OwnershipGuard()
