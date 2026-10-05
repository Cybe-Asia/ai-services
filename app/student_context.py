"""Student identity is projected by Learning/Auth/SIS, never by the AI request."""

import re
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException

from app.config import Settings
from app.learning_client import learning_data

IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


def identifier(value: object) -> bool:
    return isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None


@dataclass(frozen=True)
class StudentContext:
    actor_id: str
    student_id: str
    school_id: str
    tenant_id: str
    class_ids: tuple[str, ...]


async def resolve_student_context(
    settings: Settings, authorization: Optional[str], class_id: Optional[str] = None
) -> StudentContext:
    data = await learning_data(settings, authorization, "/api/v1/learning/context")
    actor = data.get("actor")
    if data.get("role") != "student" or not isinstance(actor, dict):
        raise HTTPException(403, "Student access required")
    if actor.get("role") != "student" or not identifier(actor.get("id")):
        raise HTTPException(403, "Student access required")
    if not identifier(data.get("school_id")) or not identifier(data.get("tenant_id")):
        raise HTTPException(503, "Student context unavailable")
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes or len(classes) > 1000:
        raise HTTPException(403, "Current Student enrollment required")
    class_ids: list[str] = []
    students: set[str] = set()
    for item in classes:
        if not isinstance(item, dict) or not identifier(item.get("id")):
            raise HTTPException(503, "Student context unavailable")
        members = item.get("students")
        if not isinstance(members, list) or len(members) != 1:
            raise HTTPException(403, "Student binding unavailable")
        if not isinstance(members[0], dict) or not identifier(members[0].get("id")):
            raise HTTPException(503, "Student context unavailable")
        students.add(members[0]["id"])
        class_ids.append(item["id"])
    if len(students) != 1 or len(set(class_ids)) != len(class_ids):
        raise HTTPException(403, "Student binding unavailable")
    if class_id is not None and class_id not in class_ids:
        raise HTTPException(403, "Current Student class required")
    return StudentContext(
        actor["id"], next(iter(students)), data["school_id"], data["tenant_id"], tuple(class_ids)
    )
