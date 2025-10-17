from pydantic import BaseModel


class config(BaseModel):
    project_name: str = "YoChameleon"
    model_id: str 