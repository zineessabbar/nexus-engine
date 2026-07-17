from typing import List
from typing import TypedDict,Annotated


class AgentState(TypedDict):
    description_projet:str
    domaines_identifies:List[str]
    textes_normatifs:List[str]
    rapport_final:str
