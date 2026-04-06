from pydantic import BaseModel, Field


class UpdateProfileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class DeleteAccountRequest(BaseModel):
    confirmation: str = Field(
        min_length=1,
        description="Must exactly match the user's email address to confirm deletion.",
    )
