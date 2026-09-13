from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=10)