import json
import logging
from time import time
from typing import Annotated, List, Literal, Optional, Dict, Any, Tuple, Union
from pydantic import AfterValidator, BaseModel, Field, computed_field, field_validator, model_validator
from enum import StrEnum
import os


class OpenAIRoles(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: OpenAIRoles = Field(
        ..., description="Role of the message: system, user, assistant")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(
        None,
        description="Optional name of the user. Valido solo se role='user'")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(
        default=None, description="ID del modello, es. gpt-3.5-turbo")
    messages: List[ChatMessage] = Field(
        ..., description="Lista di messaggi che compongono la conversazione")
    stream: Optional[bool] = Field(
        False,
        description="Se True, le risposte verranno inviate tramite streaming")
    logprobs: Optional[int] = Field(
        None,
        ge=0,
        description=
        "Se un intero, restituisce la logprobs per quel numero di token considerati"
    )
    echo: Optional[bool] = Field(
        False,
        description="Se True, il prompt viene restituito insieme alla risposta"
    )


class CompletionRequest(BaseModel):
    model: Optional[str] = Field(
        default=None, description="ID del modello, es. text-davinci-003")
    prompt: Union[str, List[str]] = Field(
        ...,
        description=
        "Il prompt da completare. Può essere una stringa o una lista di stringhe"
    )
    stream: Optional[bool] = Field(
        False,
        description="Se True, le risposte verranno inviate tramite streaming")
    logprobs: Optional[int] = Field(
        None,
        ge=0,
        description=
        "Se un intero, restituisce la logprobs per quel numero di token considerati"
    )
    echo: Optional[bool] = Field(
        False,
        description="Se True, il prompt viene restituito insieme alla risposta"
    )


class CompletionChoice(BaseModel):
    text: str = Field(..., description="Testo generato dalla completamento")
    index: int = Field(..., description="Indice della scelta nella lista")
    logprobs: Optional[Dict[str, Any]] = Field(
        None, description="Informazioni sui log probabilities se richiesto")
    finish_reason: Optional[Literal["stop", "length"]] = Field(
        None,
        description=
        "Motivo per cui la generazione si è fermata, ad esempio 'stop', 'length', ecc."
    )


class CompletionUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Numero di token del prompt")
    completion_tokens: int = Field(..., description="Numero di token generati")
    total_tokens: int = Field(
        ..., description="Totale dei token usati nella chiamata")


class CompletionResponse(BaseModel):
    id: str = Field(..., description="Identificativo univoco della richiesta")
    object: str = Field(
        default="text_completion",
        description="Tipo di oggetto restituito, es. 'text_completion'")
    created: int = Field(default_factory=time,
                         description="Timestamp di creazione della risposta")
    model: str = Field(
        ..., description="Modello utilizzato per generare la risposta")
    choices: List[CompletionChoice] = Field(
        ..., description="Elenco delle possibili completions generate")
    usage: Optional[CompletionUsage] = Field(
        None, description="Dati sull'utilizzo dei token")


class ChatChoice(BaseModel):
    index: int = Field(..., description="Indice della scelta nella lista")
    message: ChatMessage = Field(..., description="Messaggio generato")
    finish_reason: Optional[str] = Field(
        None,
        description="Motivo per cui la generazione si è conclusa, ad es. 'stop'"
    )


class ChatUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Numero di token del prompt")
    completion_tokens: int = Field(
        ..., description="Numero di token generati per la risposta")
    total_tokens: int = Field(..., description="Totale dei token utilizzati")


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Identificativo univoco della risposta")
    object: str = Field(
        ..., description="Tipo di oggetto restituito, es. 'chat.completion'")
    created: int = Field(default_factory=time,
                         description="Timestamp di creazione della risposta")
    model: str = Field(
        ..., description="Modello utilizzato per generare la risposta")
    choices: List[ChatChoice] = Field(
        ..., description="Lista delle possibili risposte generate")
    usage: Optional[ChatUsage] = Field(
        None, description="Dati sull'utilizzo dei token della risposta")


class ChatDelta(BaseModel):
    role: Optional[str] = Field(
        None, description="Ruolo del messaggio, es. 'assistant'")
    content: Optional[str] = Field(
        None, description="Porzione del contenuto generato in questo chunk")


class ChatStreamChoice(BaseModel):
    delta: ChatDelta = Field(
        ..., description="Aggiornamento parziale della risposta")
    index: int = Field(..., description="Indice della scelta")
    finish_reason: Optional[str] = Field(
        None,
        description=
        "Motivo per cui la generazione è terminata; solitamente None nei chunk parziali"
    )


class ChatCompletionChunk(BaseModel):
    id: str = Field(..., description="Identificativo univoco della richiesta")
    object: str = Field(
        ...,
        description="Di solito 'chat.completion.chunk' per i chunk in streaming"
    )
    created: int = Field(default_factory=time,
                         description="Timestamp di creazione del chunk")
    model: str = Field(
        ..., description="Modello utilizzato per generare la risposta")
    choices: List[ChatStreamChoice] = Field(
        ..., description="Lista dei chunk parziali generati")


class OpenAIErrorDetail(BaseModel):
    message: str = Field(..., description="Descrizione dell'errore")
    type: Optional[str] = Field(
        None, description="Tipo di errore, ad es. 'invalid_request_error'")
    param: Optional[Any] = Field(
        None, description="Parametro che ha causato l'errore, se applicabile")
    code: Optional[str] = Field(None, description="Codice di errore specifico")


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorDetail = Field(
        ..., description="Dettagli dell'errore restituito da OpenAI")


class Model(BaseModel):
    id: str = Field(..., description="ID del modello (es. 'text-davinci-003')")
    object: str = Field(..., description="Tipo di oggetto (es. 'model')")


class ModelsResponse(BaseModel):
    object: str = Field(
        ..., description="Tipo di oggetto della risposta (es. 'list')")
    data: List[Model] = Field(..., description="Lista dei modelli disponibili")


# Richiesta compatibile con OpenAI
class EmbeddingRequest(BaseModel):
    model: str
    input: List[str]


# Oggetto singolo di embedding
class EmbeddingData(BaseModel):
    object: str
    embedding: List[float]
    index: int


# Risposta compatibile con OpenAI
class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict  # Opzionale, per monitorare token usati


def path_validate(path: str):
    if os.path.exists(path):
        return path
    logging.error(f"Path not found at {path}")
    raise ValueError(f"Model not found at {path}")


class EngineParams(BaseModel):
    path_tokenizer_config: Annotated[str,
                                     AfterValidator(path_validate)] = Field(
                                         exclude=True)
    model_path: Annotated[str, AfterValidator(path_validate)]
    name: Optional[str] = Field (None)
    llm_params: Dict[str, Any]
    # pydantic private attribute to indicate if the model has thinking capability
    has_thinking: bool = Field(True, exclude=True)

    @model_validator(mode="after")
    def check_model(self):
        # se nome non è stato passato, lo calcolo da path
        if self.name is None:
            self.name = self.model_path.split('/')[-1].replace('.rkllm', '')
        return self

    # post validation to check if path are ok
    @computed_field  # type:ignore
    @property  # type:ignore
    def tokenizer_config(self) -> Dict[str, Any]:
        with open(self.path_tokenizer_config) as fp:
            return json.load(fp)

    # method to get the names of the models based

    def get_names(self) -> List[Tuple[str, bool]]:
        """_summary_

        :return: name + is_thinking
        :rtype: List[Tuple[str, bool]]
        """
        return [(f"{th}{self.name}",
                 th == "think-")
                for th in (("std-", "think-") if self.has_thinking else ("", ))
                ]


class AppConfig(BaseModel):
    model_collection: List[EngineParams]
