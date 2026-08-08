class FusionAgent:
    """
    Decision agent.

    Face evidence has greater weight because it comes from
    the trained visual deepfake detector.
    """

    FACE_WEIGHT = 0.75
    AUDIO_WEIGHT = 0.25

    def analyze(self, face_result, audio_result):

        face_prediction = face_result.get(
            "prediction",
            "Unknown"
        )

        face_confidence = float(
            face_result.get("confidence", 0.0)
        )

        audio_prediction = audio_result.get(
            "prediction",
            "Unknown"
        )

        audio_confidence = float(
            audio_result.get("confidence", 0.0)
        )

        face_fake_probability = self._to_fake_probability(
            face_prediction,
            face_confidence
        )

        audio_fake_probability = self._to_fake_probability(
            audio_prediction,
            audio_confidence
        )

        # Clamp all probabilities to prevent malformed agent
        # output from propagating through the fusion stage.
        if face_fake_probability is not None:
            face_fake_probability = max(
                0.0,
                min(1.0, face_fake_probability)
            )

        if audio_fake_probability is not None:
            audio_fake_probability = max(
                0.0,
                min(1.0, audio_fake_probability)
            )

        # --------------------------------------------------
        # No usable evidence
        # --------------------------------------------------

        if (
            face_fake_probability is None
            and audio_fake_probability is None
        ):
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "fake_probability": None,
                "explanation": (
                    "Neither the Face Agent nor the Audio Agent "
                    "provided usable evidence."
                ),
                "evidence": {
                    "face_fake_probability": None,
                    "audio_fake_probability": None
                }
            }

        # --------------------------------------------------
        # Face only
        # --------------------------------------------------

        if face_fake_probability is not None and audio_fake_probability is None:

            final_fake_probability = face_fake_probability

            explanation = (
                f"Audio Agent could not provide usable evidence. "
                f"The final decision therefore relies on the "
                f"Face Agent, which predicted {face_prediction} "
                f"with {face_confidence * 100:.1f}% confidence."
            )

        # --------------------------------------------------
        # Audio only
        # --------------------------------------------------

        elif face_fake_probability is None and audio_fake_probability is not None:

            final_fake_probability = audio_fake_probability

            explanation = (
                f"Face Agent could not provide usable visual evidence. "
                f"The final decision relies on the Audio Agent, "
                f"which predicted {audio_prediction} with "
                f"{audio_confidence * 100:.1f}% confidence. "
                f"This result should be treated cautiously because "
                f"the audio agent uses lightweight heuristics."
            )

        # --------------------------------------------------
        # Both agents
        # --------------------------------------------------

        else:

            final_fake_probability = (
                self.FACE_WEIGHT * face_fake_probability
                + self.AUDIO_WEIGHT * audio_fake_probability
            )

            conflicting = (
                face_prediction in ("Real", "Fake")
                and audio_prediction in ("Real", "Fake")
                and face_prediction != audio_prediction
            )

            if conflicting:

                explanation = (
                    f"Face Agent detected {face_prediction} evidence "
                    f"with {face_confidence * 100:.1f}% confidence, "
                    f"while Audio Agent detected {audio_prediction} "
                    f"evidence with {audio_confidence * 100:.1f}% "
                    f"confidence. The agents therefore disagree. "
                    f"The Fusion Agent resolves this conflict using "
                    f"{self.FACE_WEIGHT:.0%} visual and "
                    f"{self.AUDIO_WEIGHT:.0%} audio weighting, "
                    f"giving greater importance to the trained "
                    f"visual detector."
                )

            else:

                explanation = (
                    f"Face Agent predicted {face_prediction} with "
                    f"{face_confidence * 100:.1f}% confidence, and "
                    f"Audio Agent predicted {audio_prediction} with "
                    f"{audio_confidence * 100:.1f}% confidence. "
                    f"The Fusion Agent combines both signals using "
                    f"{self.FACE_WEIGHT:.0%} visual and "
                    f"{self.AUDIO_WEIGHT:.0%} audio weighting."
                )

        final_fake_probability = max(
            0.0,
            min(1.0, final_fake_probability)
        )

        prediction = (
            "Fake"
            if final_fake_probability >= 0.5
            else "Real"
        )

        confidence = max(
            final_fake_probability,
            1.0 - final_fake_probability
        )

        return {
            "prediction": prediction,
            "confidence": float(confidence * 100.0),
            "fake_probability": float(final_fake_probability),
            "explanation": explanation,
            "evidence": {
                "face_fake_probability": face_fake_probability,
                "audio_fake_probability": audio_fake_probability
            }
        }

    @staticmethod
    def _to_fake_probability(
        prediction,
        confidence
    ):
        """
        Convert agent prediction + confidence into fake probability.

        Fake + 0.90 -> 0.90
        Real + 0.80 -> 0.20
        Unknown       -> None
        """

        confidence = max(
            0.0,
            min(1.0, float(confidence))
        )

        if prediction == "Fake":
            return confidence

        if prediction == "Real":
            return 1.0 - confidence

        return None