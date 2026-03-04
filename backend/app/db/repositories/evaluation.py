"""Repositories for eval items, evaluations, and judge outputs."""
from typing import Dict, List, Optional, Set

from sqlalchemy import func
from sqlalchemy.orm import Session

from ...models.database import (
    EvalItemModel,
    EvaluationModel,
    JudgeOutputModel,
    json_deserializer,
    json_serializer,
)
from ...models.schemas import (
    ClassificationResult,
    DatasetSplit,
    EvalItemCreate,
    EvalItemInDB,
    EvaluationCreate,
    EvaluationInDB,
    JudgeOutput,
    JudgeOutputInDB,
)
from .common import generate_id


class EvalItemRepository:
    """Repository for eval item operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, item_id: str) -> Optional[EvalItemInDB]:
        model = self.db.query(EvalItemModel).filter(EvalItemModel.id == item_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_by_external_id(self, external_id: str) -> Optional[EvalItemInDB]:
        model = self.db.query(EvalItemModel).filter(EvalItemModel.external_id == external_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_all(
        self,
        split: Optional[DatasetSplit] = None,
        scenario_id: Optional[str] = None,
        candidate_source: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[List[EvalItemInDB], int]:
        query = self.db.query(EvalItemModel)
        if split:
            query = query.filter(EvalItemModel.split == split)
        if scenario_id:
            query = query.filter(EvalItemModel.scenario_id == scenario_id)
        if candidate_source:
            query = query.filter(EvalItemModel.candidate_source == candidate_source)

        total = query.count()
        models = (
            query.order_by(EvalItemModel.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return [self._to_schema(m) for m in models], total

    def get_by_split(self, split: DatasetSplit) -> List[EvalItemInDB]:
        models = (
            self.db.query(EvalItemModel)
            .filter(EvalItemModel.split == split)
            .order_by(EvalItemModel.created_at)
            .all()
        )
        return [self._to_schema(m) for m in models]

    def create(self, item: EvalItemCreate, commit: bool = True) -> EvalItemInDB:
        model = EvalItemModel(
            id=generate_id(),
            external_id=item.external_id,
            split=item.split,
            system_prompt=item.system_prompt,
            question=item.question,
            response=item.response,
            metadata_json=json_serializer(item.metadata) if item.metadata else None,
            scenario_id=item.scenario_id,
            candidate_source=item.candidate_source,
        )
        self.db.add(model)
        if commit:
            self.db.commit()
            self.db.refresh(model)
        return self._to_schema(model)

    def create_batch(self, items: List[EvalItemCreate], skip_duplicates: bool = True) -> int:
        """Batch create eval items in a single transaction.

        When skip_duplicates=True, items with an external_id that already exists
        in the database are silently skipped.
        """
        existing_ids: Set[str] = set()
        seen_ids: Set[str] = set()
        if skip_duplicates:
            ext_ids = [i.external_id for i in items if i.external_id]
            if ext_ids:
                rows = self.db.query(EvalItemModel.external_id).filter(
                    EvalItemModel.external_id.in_(ext_ids)
                ).all()
                existing_ids = {r[0] for r in rows}
                seen_ids.update(existing_ids)

        models = []
        for item in items:
            if skip_duplicates and item.external_id:
                if item.external_id in seen_ids:
                    continue
                seen_ids.add(item.external_id)
            model = EvalItemModel(
                id=generate_id(),
                external_id=item.external_id,
                split=item.split,
                system_prompt=item.system_prompt,
                question=item.question,
                response=item.response,
                metadata_json=json_serializer(item.metadata) if item.metadata else None,
                scenario_id=item.scenario_id,
                candidate_source=item.candidate_source,
            )
            models.append(model)
        if models:
            self.db.add_all(models)
            self.db.commit()
        return len(models)

    def flush(self):
        """Commit pending changes."""
        self.db.commit()

    def update_masked(self, item_id: str, masked_text: str, commit: bool = True) -> Optional[EvalItemInDB]:
        model = self.db.query(EvalItemModel).filter(EvalItemModel.id == item_id).first()
        if not model:
            return None
        model.masked_text = masked_text
        if commit:
            self.db.commit()
            self.db.refresh(model)
        return self._to_schema(model)

    def get_by_scenario(self, scenario_id: str) -> List[EvalItemInDB]:
        """Get all candidate items for a given scenario."""
        models = self.db.query(EvalItemModel).filter(EvalItemModel.scenario_id == scenario_id).all()
        return [self._to_schema(m) for m in models]

    def _to_schema(self, model: EvalItemModel) -> EvalItemInDB:
        return EvalItemInDB(
            id=model.id,
            external_id=model.external_id,
            split=model.split,
            system_prompt=model.system_prompt,
            question=model.question,
            response=model.response,
            metadata=model.item_metadata,
            scenario_id=model.scenario_id,
            candidate_source=model.candidate_source,
            masked_text=model.masked_text,
            created_at=model.created_at,
        )

    def get_summary_stats(self, split: DatasetSplit) -> Dict:
        """Get aggregated evaluation stats for a split using JOINed queries."""
        total_evals = (
            self.db.query(func.count(EvaluationModel.id))
            .join(EvalItemModel, EvalItemModel.id == EvaluationModel.item_id)
            .filter(EvalItemModel.split == split)
            .scalar()
            or 0
        )

        if total_evals == 0:
            return {
                "total_evaluations": 0,
                "gate_fail_count": 0,
                "scores_raw": [],
                "tags_raw": [],
            }

        judge_rows = (
            self.db.query(
                JudgeOutputModel.gates_json,
                JudgeOutputModel.scores_json,
                JudgeOutputModel.failure_tags_json,
            )
            .join(EvaluationModel, EvaluationModel.id == JudgeOutputModel.evaluation_id)
            .join(EvalItemModel, EvalItemModel.id == EvaluationModel.item_id)
            .filter(EvalItemModel.split == split)
            .all()
        )

        gate_fails = 0
        score_sums: Dict[str, float] = {}
        score_counts: Dict[str, int] = {}
        tag_counts: Dict[str, int] = {}

        for gates_json, scores_json, tags_json in judge_rows:
            gates = json_deserializer(gates_json) or []
            scores = json_deserializer(scores_json) or []
            tags = json_deserializer(tags_json) or []

            if not all(g.get("passed", True) for g in gates):
                gate_fails += 1
            for s in scores:
                st = s.get("score_type", "")
                sv = s.get("score", 0)
                score_sums[st] = score_sums.get(st, 0) + sv
                score_counts[st] = score_counts.get(st, 0) + 1
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        avg_scores = {
            k: score_sums[k] / score_counts[k]
            for k in score_sums
            if score_counts[k] > 0
        }

        return {
            "total_evaluations": total_evals,
            "gate_fail_count": gate_fails,
            "avg_scores": avg_scores,
            "tag_counts": tag_counts,
        }


class EvaluationRepository:
    """Repository for evaluation operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, eval_id: str) -> Optional[EvaluationInDB]:
        model = self.db.query(EvaluationModel).filter(EvaluationModel.id == eval_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_by_item(self, item_id: str) -> List[EvaluationInDB]:
        models = (
            self.db.query(EvaluationModel)
            .filter(EvaluationModel.item_id == item_id)
            .order_by(EvaluationModel.created_at.desc())
            .all()
        )
        return [self._to_schema(m) for m in models]

    def get_by_version(
        self,
        prompt_version: str,
        model_version: str,
        docs_version: str,
    ) -> List[EvaluationInDB]:
        models = self.db.query(EvaluationModel).filter(
            EvaluationModel.prompt_version == prompt_version,
            EvaluationModel.model_version == model_version,
            EvaluationModel.docs_version == docs_version,
        ).all()
        return [self._to_schema(m) for m in models]

    def get_evaluated_item_ids(
        self,
        prompt_version: str,
        model_version: str,
        docs_version: str,
    ) -> set:
        """Return set of item_ids that already have an evaluation for this version triple."""
        rows = self.db.query(EvaluationModel.item_id).filter(
            EvaluationModel.prompt_version == prompt_version,
            EvaluationModel.model_version == model_version,
            EvaluationModel.docs_version == docs_version,
        ).all()
        return {r[0] for r in rows}

    def create(
        self,
        evaluation: EvaluationCreate,
        trace_id: Optional[str] = None,
        commit: bool = True,
    ) -> EvaluationInDB:
        model = EvaluationModel(
            id=generate_id(),
            item_id=evaluation.item_id,
            prompt_version=evaluation.prompt_version,
            model_version=evaluation.model_version,
            docs_version=evaluation.docs_version,
            trace_id=trace_id,
        )
        self.db.add(model)
        if commit:
            self.db.commit()
            self.db.refresh(model)
        else:
            self.db.flush()
        return self._to_schema(model)

    def update_classification(
        self,
        eval_id: str,
        classification: ClassificationResult,
        commit: bool = True,
    ) -> Optional[EvaluationInDB]:
        model = self.db.query(EvaluationModel).filter(EvaluationModel.id == eval_id).first()
        if not model:
            return None
        model.classification_json = json_serializer(classification.model_dump())
        if commit:
            self.db.commit()
            self.db.refresh(model)
        return self._to_schema(model)

    def _to_schema(self, model: EvaluationModel) -> EvaluationInDB:
        classification = None
        if model.classification:
            classification = ClassificationResult(**model.classification)

        judge = None
        if model.judge_output:
            judge = JudgeOutput(
                gates=model.judge_output.gates,
                scores=model.judge_output.scores,
                failure_tags=model.judge_output.failure_tags,
                summary_of_issue=model.judge_output.summary_of_issue,
                what_to_fix=model.judge_output.what_to_fix,
                rag_citations=model.judge_output.rag_citations,
            )

        return EvaluationInDB(
            id=model.id,
            item_id=model.item_id,
            prompt_version=model.prompt_version,
            model_version=model.model_version,
            docs_version=model.docs_version,
            classification=classification,
            judge_output=judge,
            trace_id=model.trace_id,
            created_at=model.created_at,
        )


class JudgeOutputRepository:
    """Repository for judge output operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, output_id: str) -> Optional[JudgeOutputInDB]:
        model = self.db.query(JudgeOutputModel).filter(JudgeOutputModel.id == output_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_by_evaluation(self, eval_id: str) -> Optional[JudgeOutputInDB]:
        model = self.db.query(JudgeOutputModel).filter(JudgeOutputModel.evaluation_id == eval_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def create(self, eval_id: str, output: JudgeOutput, commit: bool = True) -> JudgeOutputInDB:
        model = JudgeOutputModel(
            id=generate_id(),
            evaluation_id=eval_id,
            gates_json=json_serializer([g.model_dump() for g in output.gates]),
            scores_json=json_serializer([s.model_dump() for s in output.scores]),
            failure_tags_json=json_serializer(output.failure_tags),
            summary_of_issue=output.summary_of_issue,
            what_to_fix=output.what_to_fix,
            rag_citations_json=json_serializer(output.rag_citations),
        )
        self.db.add(model)
        if commit:
            self.db.commit()
            self.db.refresh(model)
        else:
            self.db.flush()
        return self._to_schema(model)

    def _to_schema(self, model: JudgeOutputModel) -> JudgeOutputInDB:
        return JudgeOutputInDB(
            id=model.id,
            evaluation_id=model.evaluation_id,
            gates=[g for g in model.gates],
            scores=[s for s in model.scores],
            failure_tags=model.failure_tags,
            summary_of_issue=model.summary_of_issue,
            what_to_fix=model.what_to_fix,
            rag_citations=model.rag_citations,
            created_at=model.created_at,
        )
