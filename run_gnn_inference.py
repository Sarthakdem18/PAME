from src.gnn.infer_gnn import infer
df = infer("data/aux_hate/task_a/Test_Task_A.xlsx")
df.to_excel("gnn_predictions.xlsx", index=False)
print("Inference complete.")
