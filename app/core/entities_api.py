from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import StrEnum


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
    finish_reason: Optional[str] = Field(
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
        ..., description="Tipo di oggetto restituito, es. 'text_completion'")
    created: int = Field(...,
                         description="Timestamp di creazione della risposta")
    model: str = Field(
        ..., description="Modello utilizzato per generare la risposta")
    choices: List[CompletionChoice] = Field(
        ..., description="Elenco delle possibili completions generate")
    usage: Optional[CompletionUsage] = Field(
        None, description="Dati sull'utilizzo dei token")


class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        description="Ruolo del messaggio, es. 'system', 'user', 'assistant'")
    content: str = Field(..., description="Contenuto del messaggio")


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
    created: int = Field(...,
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
    created: int = Field(..., description="Timestamp di creazione del chunk")
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
    object: str = Field(..., description="Tipo di oggetto della risposta (es. 'list')")
    data: List[Model] = Field(..., description="Lista dei modelli disponibili")
