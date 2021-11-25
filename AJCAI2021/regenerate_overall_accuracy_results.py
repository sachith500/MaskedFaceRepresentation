from AJCAI2021.pipeline import Pipeline

test_types = ['sex', 'race', 'age_regression', 'age_classification']
results = []
for test_type in test_types:
    pipeline = Pipeline(test_type)
    pipeline_results = pipeline.process()
    result = [test_type, pipeline_results[0]]
    results.append(result)

print(f"Type\t Masked Face (uniform split)")
print(f"==========================================================")
for result in results:
    print(f"{result[0]}\t {result[1][1]}")
