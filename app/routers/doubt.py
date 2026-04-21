"""
Doubt Router
============
POST /api/v1/doubt/solve
  Input:  topic, doubt question, optional context
  Output: detailed AI answer with related concepts & follow-ups
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import APIResponse, DoubtAnswer, DoubtRequest
from app.services.llm_service import llm_service
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


@router.post(
    "/solve",
    response_model=APIResponse,
    summary="Solve a student's doubt",
    description=(
        "Pass the topic and the specific doubt. AI returns a detailed "
        "explanation with examples, related concepts, and follow-up questions."
    ),
)
async def solve_doubt(request: DoubtRequest):
    logger.info(
        f"[Doubt] topic='{request.topic}', "
        f"doubt='{request.doubt[:60]}...'"
    )

    try:
        answer_data = await llm_service.solve_doubt(
            topic        = request.topic,
            doubt        = request.doubt,
            context      = request.context,
            student_name = request.student_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    try:
        answer = DoubtAnswer(
            topic                = answer_data.get("topic", request.topic),
            doubt                = answer_data.get("doubt", request.doubt),
            answer               = answer_data.get("answer", ""),
            related_concepts     = answer_data.get("related_concepts", []),
            follow_up_questions  = answer_data.get("follow_up_questions", []),
            answered_at          = datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        logger.error(f"[Doubt] Schema validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Answer generated but response formatting failed. Please retry."
        )

    return APIResponse(
        success = True,
        message = f"Doubt solved for topic: '{request.topic}'",
        data    = answer.dict(),
    )
