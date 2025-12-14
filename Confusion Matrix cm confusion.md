# Confusion Matrix
cm = confusion_matrix(gts, preds)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - PyTorch Model')
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('artifacts/confusion_matrix_pytorch.png', dpi=300, bbox_inches='tight')
plt.show()
