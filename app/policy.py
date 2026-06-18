from app.schemas import ActorRole

SENSITIVE_MARKERS = (
    "best student",
    "top student",
    "ranking",
    "rank",
    "nilai",
    "score",
    "grade",
    "payment",
    "pembayaran",
    "tagihan",
    "invoice",
    "eoi",
    "email",
    "lead",
    "leads",
    "applicant",
    "applicants",
    "pendaftar",
    "terdaftar",
    "daftar",
    "mendaftar",
    "orang yang daftar",
    "calon siswa",
    "calon murid",
    "calon keluarga",
    "registration",
    "registered",
    "enquiry",
    "enquiries",
    "inquiry",
    "inquiries",
    "student",
    "siswa",
)

PRIVILEGED_ROLES = {ActorRole.owner, ActorRole.admin, ActorRole.teacher}


def requires_privileged_role(message: str) -> bool:
    lowered = message.casefold()
    return any(marker in lowered for marker in SENSITIVE_MARKERS)


def is_allowed_for_message(role: ActorRole, message: str) -> bool:
    if not requires_privileged_role(message):
        return True
    return role in PRIVILEGED_ROLES
