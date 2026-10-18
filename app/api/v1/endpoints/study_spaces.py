from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models.study_space import StudySpace
from app.models.user import User
from app.schemas.study_space import (
    StudySpaceCreate,
    StudySpaceResponse,
    StudySpaceUpdate,
)

router = APIRouter(
    prefix="/study-spaces",
    tags=["Study Spaces"],
)


def get_owned_study_space(
    space_id: UUID,
    user_id: UUID,
    db: Session,
) -> StudySpace:
    """Return a study space only if it belongs to the current user."""
    study_space = (
        db.query(StudySpace)
        .filter(
            StudySpace.id == space_id,
            StudySpace.user_id == user_id,
        )
        .first()
    )

    if study_space is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study space not found.",
        )

    return study_space


@router.post(
    "/",
    response_model=StudySpaceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_study_space(
    payload: StudySpaceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    study_space = StudySpace(
        user_id=current_user.id,
        name=payload.name.strip(),
        description=payload.description,
    )

    if not study_space.name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Study space name cannot be blank.",
        )

    db.add(study_space)
    db.commit()
    db.refresh(study_space)

    return study_space


@router.get(
    "/",
    response_model=list[StudySpaceResponse],
)
def list_study_spaces(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(StudySpace)
        .filter(StudySpace.user_id == current_user.id)
        .order_by(StudySpace.created_at.desc())
        .all()
    )


@router.get(
    "/{space_id}",
    response_model=StudySpaceResponse,
)
def get_study_space(
    space_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_owned_study_space(
        space_id=space_id,
        user_id=current_user.id,
        db=db,
    )


@router.patch(
    "/{space_id}",
    response_model=StudySpaceResponse,
)
def update_study_space(
    space_id: UUID,
    payload: StudySpaceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    study_space = get_owned_study_space(
        space_id=space_id,
        user_id=current_user.id,
        db=db,
    )

    changes = payload.model_dump(exclude_unset=True)

    if not changes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide at least one field to update.",
        )

    if "name" in changes:
        if changes["name"] is None or not changes["name"].strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Study space name cannot be blank.",
            )
        changes["name"] = changes["name"].strip()

    for field, value in changes.items():
        setattr(study_space, field, value)

    db.commit()
    db.refresh(study_space)

    return study_space


@router.delete(
    "/{space_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_study_space(
    space_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    study_space = get_owned_study_space(
        space_id=space_id,
        user_id=current_user.id,
        db=db,
    )

    db.delete(study_space)
    db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)
