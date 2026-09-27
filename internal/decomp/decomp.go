package decomp

func Recall(pEscalation, pAlert, pHuman, reviewerRecall float64) float64 {
	return pEscalation * (pAlert + pHuman*reviewerRecall)
}
