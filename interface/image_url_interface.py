from pydantic import BaseModel


class ImageUrlInterface(BaseModel):
    url: str
