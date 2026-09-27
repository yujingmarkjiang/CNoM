package cost

func CameraMonth(edgeFixed, cloudFull, uploadFrac float64) (hs, cloud, ratio float64) {
	hs = edgeFixed + cloudFull*uploadFrac
	cloud = cloudFull
	if hs > 0 {
		ratio = cloud / hs
	}
	return
}
