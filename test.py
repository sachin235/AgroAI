from wheat_quality_predictor import predict

good, not_good = predict('./static/juvd.jpg')

print(good, not_good)