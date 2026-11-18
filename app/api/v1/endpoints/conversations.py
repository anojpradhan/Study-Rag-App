from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models.conversation import Conversation
from app.models.study_space import StudySpace
from app.models.user import User
from app.schemas.conversation import (
    ConversationCreate,
    ConversationDetailResponse,
    ConversationResponse,
    ConversationUpdate,
)

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


def get_owned_conversation(
    conversation_id: UUID,
    user_id: UUID,
    db: Session,
) -> Conversation:
    """Return a conversation only if it belongs to the current user."""

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        .first()
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return conversation


@router.post(
    "/",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    payload: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a conversation inside an owned Study Space."""

    study_space = (
        db.query(StudySpace)
        .filter(
            StudySpace.id == payload.study_space_id,
            StudySpace.user_id == current_user.id,
        )
        .first()
    )

    if study_space is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study space not found.",
        )

    conversation = Conversation(
        user_id=current_user.id,
        study_space_id=study_space.id,
        title=payload.title.strip(),
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation


@router.get(
    "/study-space/{study_space_id}",
    response_model=list[ConversationResponse],
)
def list_conversations(
    study_space_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List conversations belonging to an owned Study Space."""

    study_space = (
        db.query(StudySpace)
        .filter(
            StudySpace.id == study_space_id,
            StudySpace.user_id == current_user.id,
        )
        .first()
    )

    if study_space is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study space not found.",
        )

    return (
        db.query(Conversation)
        .filter(
            Conversation.study_space_id == study_space_id,
            Conversation.user_id == current_user.id,
        )
        .order_by(Conversation.updated_at.desc())
        .all()
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
)
def get_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a conversation with its messages."""

    conversation = get_owned_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
        db=db,
    )

    return conversation


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
def update_conversation(
    conversation_id: UUID,
    payload: ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Rename an owned conversation."""

    conversation = get_owned_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
        db=db,
    )

    conversation.title = payload.title.strip()

    if not conversation.title:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Conversation title cannot be blank.",
        )

    db.commit()
    db.refresh(conversation)

    return conversation


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete an owned conversation and its messages."""

    conversation = get_owned_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
        db=db,
    )

    db.delete(conversation)
    db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)
