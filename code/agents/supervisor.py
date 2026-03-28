"""
Loan Supervisor Agent - Orchestrates the underwriting workflow
"""

from typing import Dict, Any, List, Optional
from .base import BaseAgent, ApplicationData, DecisionResult
import logging

logger = logging.getLogger(__name__)


class LoanSupervisor(BaseAgent):
    """
    Supervisor agent that orchestrates the entire underwriting workflow.
    Delegates tasks to specialized agents and makes final decisions.
    """

    def __init__(
        self, agent_id: str = "supervisor", config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_id, config)
        self.decision_threshold_approve = self.config.get("threshold_approve", 0.7)
        self.decision_threshold_deny = self.config.get("threshold_deny", 0.3)
        self.human_review_threshold = self.config.get("human_review_threshold", 0.5)
        self.max_negotiation_rounds = self.config.get("max_negotiation_rounds", 3)

    def process(self, application: ApplicationData) -> DecisionResult:
        """
        Main orchestration logic for processing a loan application.

        Steps:
        1. Parse documents
        2. Verify income and employment
        3. Calculate credit score
        4. Assess fraud risk
        5. Check fairness constraints
        6. Generate explanation
        7. Make final decision with human review gating if needed
        """
        self.logger.info(f"Processing application: {application.application_id}")

        agent_outputs = {}
        evidence = []

        # Step 1: Document processing (delegated to document processor)
        doc_result = self._delegate_document_processing(application)
        agent_outputs["document_processor"] = doc_result
        evidence.append(
            {
                "agent": "document_processor",
                "confidence": doc_result.get("confidence", 1.0),
                "summary": doc_result.get("summary", "Documents processed"),
            }
        )

        # Step 2: Income verification (delegated to income verifier)
        income_result = self._delegate_income_verification(application, doc_result)
        agent_outputs["income_verifier"] = income_result
        evidence.append(
            {
                "agent": "income_verifier",
                "confidence": income_result.get("confidence", 1.0),
                "verified": income_result.get("verified", False),
            }
        )

        # Step 3: Credit scoring (delegated to credit scoring agent)
        credit_result = self._delegate_credit_scoring(
            application, doc_result, income_result
        )
        agent_outputs["credit_scorer"] = credit_result
        evidence.append(
            {
                "agent": "credit_scorer",
                "score": credit_result.get("credit_score", 0),
                "pd": credit_result.get("probability_default", 0.5),
            }
        )

        # Step 4: Fraud & Risk assessment (delegated to fraud agent)
        fraud_result = self._delegate_fraud_detection(
            application, doc_result, income_result
        )
        agent_outputs["fraud_detector"] = fraud_result
        evidence.append(
            {
                "agent": "fraud_detector",
                "fraud_score": fraud_result.get("fraud_score", 0),
                "alerts": fraud_result.get("alerts", []),
            }
        )

        # Step 5: Fairness check (delegated to fairness agent)
        fairness_result = self._delegate_fairness_check(application, credit_result)
        agent_outputs["fairness_checker"] = fairness_result

        # Step 6: Aggregate scores and make preliminary decision
        aggregate_score = self._aggregate_scores(
            credit_result, fraud_result, income_result
        )

        preliminary_decision = self._make_preliminary_decision(aggregate_score)
        self.logger.debug(
            f"Preliminary decision for {application.application_id}: {preliminary_decision}"
        )

        # Step 7: Negotiation loop for borderline cases
        if self._is_borderline(aggregate_score):
            final_score, negotiation_log = self._negotiation_loop(
                application, aggregate_score, agent_outputs
            )
            agent_outputs["negotiation"] = negotiation_log
        else:
            final_score = aggregate_score

        # Step 8: Final decision with human review gating
        decision, confidence = self._make_final_decision(final_score, fairness_result)

        # Step 9: Generate explanation (delegated to explanation agent)
        explanation_result = self._delegate_explanation_generation(
            application, decision, agent_outputs, evidence
        )

        # Construct final decision result
        result = DecisionResult(
            application_id=application.application_id,
            decision=decision,
            confidence=confidence,
            risk_score=1.0 - final_score,
            recommended_terms=self._generate_terms(decision, final_score),
            rationale=explanation_result.get("rationale", ""),
            evidence=evidence,
            agent_outputs=agent_outputs,
            fairness_metrics=fairness_result.get("metrics", {}),
        )

        self.logger.info(
            f"Decision for {application.application_id}: {decision} (confidence: {confidence:.3f})"
        )
        return result

    def _delegate_document_processing(
        self, application: ApplicationData
    ) -> Dict[str, Any]:
        """Delegate to DocumentProcessor agent"""
        # In full implementation, this sends message to actual agent
        # For now, mock the response with realistic structure
        return {
            "confidence": 0.95,
            "summary": "All required documents present and verified",
            "extracted_fields": {
                "income": application.financial_info.get("annual_income", 0),
                "employment_verified": True,
            },
        }

    def _delegate_income_verification(
        self, application: ApplicationData, doc_result: Dict
    ) -> Dict[str, Any]:
        """Delegate to IncomeVerifier agent"""
        return {
            "verified": True,
            "confidence": 0.92,
            "reported_income": application.financial_info.get("annual_income", 0),
            "verified_income": application.financial_info.get("annual_income", 0),
        }

    def _delegate_credit_scoring(
        self, application: ApplicationData, doc_result: Dict, income_result: Dict
    ) -> Dict[str, Any]:
        """Delegate to CreditScoring agent - returns score and PD"""
        # This will call actual model in full implementation
        return {
            "credit_score": 680,  # Placeholder - real from model
            "probability_default": 0.15,
            "features": {},
        }

    def _delegate_fraud_detection(
        self, application: ApplicationData, doc_result: Dict, income_result: Dict
    ) -> Dict[str, Any]:
        """Delegate to FraudDetector agent"""
        return {"fraud_score": 0.05, "alerts": [], "confidence": 0.88}

    def _delegate_fairness_check(
        self, application: ApplicationData, credit_result: Dict
    ) -> Dict[str, Any]:
        """Delegate to FairnessAgent"""
        return {
            "passed": True,
            "metrics": {"demographic_parity_diff": 0.02, "equalized_odds_diff": 0.03},
        }

    def _delegate_explanation_generation(
        self,
        application: ApplicationData,
        decision: str,
        agent_outputs: Dict,
        evidence: List,
    ) -> Dict[str, Any]:
        """Delegate to ExplanationAgent"""
        return {
            "rationale": f"Decision: {decision}. Based on credit score, income verification, and fraud assessment.",
            "key_factors": evidence,
        }

    def _aggregate_scores(
        self, credit_result: Dict, fraud_result: Dict, income_result: Dict
    ) -> float:
        """Aggregate scores from different agents into single score"""
        # Weighted combination
        credit_weight = 0.5
        fraud_weight = 0.3
        income_weight = 0.2

        credit_score_normalized = 1.0 - credit_result.get("probability_default", 0.5)
        fraud_score_normalized = 1.0 - fraud_result.get("fraud_score", 0.0)
        income_score = 1.0 if income_result.get("verified", False) else 0.5

        aggregate = (
            credit_score_normalized * credit_weight
            + fraud_score_normalized * fraud_weight
            + income_score * income_weight
        )

        return aggregate

    def _make_preliminary_decision(self, score: float) -> str:
        """Make preliminary decision based on aggregate score"""
        if score >= self.decision_threshold_approve:
            return "approve"
        elif score <= self.decision_threshold_deny:
            return "deny"
        else:
            return "review"

    def _is_borderline(self, score: float) -> bool:
        """
        Check if case is borderline and needs negotiation.

        """
        return self.decision_threshold_deny < score < self.decision_threshold_approve

    def _negotiation_loop(
        self, application: ApplicationData, initial_score: float, agent_outputs: Dict
    ) -> tuple:
        """Negotiation loop for borderline cases"""
        score = initial_score
        log = []

        for round_num in range(self.max_negotiation_rounds):
            # In full implementation, agents can revise their assessments
            log.append({"round": round_num + 1, "score": score, "action": "no_change"})
            # Early exit if score moves out of borderline range
            if not self._is_borderline(score):
                break

        return score, log

    def _make_final_decision(self, score: float, fairness_result: Dict) -> tuple:
        """Make final decision with human review gating"""
        confidence = abs(score - 0.5) * 2  # Convert to 0-1 confidence

        # Check if human review is needed
        if self._is_borderline(score) or not fairness_result.get("passed", True):
            decision = "review"
            confidence = min(confidence, 0.6)  # Cap confidence for review cases
        elif score >= self.decision_threshold_approve:
            decision = "approve"
        elif score <= self.decision_threshold_deny:
            decision = "deny"
        else:
            decision = "review"

        return decision, confidence

    def _generate_terms(self, decision: str, score: float) -> Optional[Dict[str, Any]]:
        """Generate recommended loan terms for approved applications"""
        if decision != "approve":
            return None

        # Risk-based pricing
        base_rate = 0.05
        risk_premium = (1.0 - score) * 0.10

        return {
            "interest_rate": base_rate + risk_premium,
            "term_months": 36,
            "max_amount": 50000 * score,
        }
