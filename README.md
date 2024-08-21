# Tenser_Go_Assignment

## Run Tenser.ipynb on google colab 

Example Visualizations

Table:
+----------------+---------------------+-------------------+-------------+

| Metric         | Before Fine-Tuning | After Fine-Tuning | Improvement |

+----------------+---------------------+-------------------+-------------+

| Accuracy       | 60.0%               | 85.0%             | +25.0%      |
| Loss           | 1.20                | 0.45              | -0.75       |

+----------------+---------------------+-------------------+-------------+




# Before Fine-Tuning: GPT-2's Pre-Trained State

1. Selection Criteria for the Base LLM

Explanation:

Base Model: GPT-2 is chosen due to its high capacity and general capabilities in natural language processing. It’s pre-trained on a large and diverse corpus of text from the internet, making it versatile for various NLP tasks.
Criteria for Selection:
General Language Understanding: GPT-2 has been trained on extensive textual data, enabling it to understand and generate coherent text across a wide range of topics.
Flexibility: It can be adapted for different tasks with fine-tuning, thanks to its large-scale pre-training.
Performance Benchmarks: GPT-2 has demonstrated strong performance in various NLP benchmarks and tasks, providing a solid foundation for further specialization.

2. Task-Specific Considerations for Fine-Tuning

Explanation:

Task: Sentiment classification of movie reviews.
Considerations:
Nature of Task: Sentiment analysis requires understanding nuanced language and sentiment expressions. GPT-2’s general training allows it to perform well, but it may not be optimal without task-specific fine-tuning.
Domain Relevance: While GPT-2 can generate text and perform tasks in a zero-shot setting, it lacks specialized knowledge in sentiment analysis unless fine-tuned on a relevant dataset.

3. Data Preparation and Preprocessing Steps

Explanation:

Dataset: IMDB dataset used for sentiment classification.
Preprocessing Steps:
Tokenization: Convert raw text into token IDs that GPT-2 can process.
Padding and Truncation: Ensure all sequences are of consistent length for model training.
Handling Labels: Convert sentiment labels (positive/negative) into a format suitable for the model. In the case of GPT-2, this might involve adapting the format for text generation tasks, as GPT-2 primarily generates text rather than classifying it directly.

4. Fine-Tuning Hyperparameters and Optimization Strategies

Explanation:

Hyperparameters:
Learning Rate: Determines how much to adjust the model weights during training.
Batch Size: Number of samples processed before updating the model weights.
Number of Epochs: Number of times the model sees the entire dataset.
Weight Decay: Regularization parameter to prevent overfitting.
Optimization Strategies:
Learning Rate Scheduling: Adjust learning rates over time to improve convergence.
Early Stopping: Stop training when performance ceases to improve on the validation set.

5. Evaluation Metrics and Performance Analysis

Explanation:

Metrics:
Loss: Measures how well the model's predictions match the actual labels. Lower loss indicates better performance.
Accuracy: Percentage of correct classifications. While GPT-2 is primarily used for generation, accuracy in classification tasks can be evaluated after fine-tuning.
Performance Analysis:
Baseline Performance: Performance of GPT-2 in a zero-shot setting or before fine-tuning, showing its initial capabilities and limitations.


# After Fine-Tuning: GPT-2 Adapted for Sentiment Classification
1. Selection Criteria for the Base LLM

Explanation:

Base Model: GPT-2 was initially chosen for its general language capabilities and versatility. After fine-tuning, the focus remains on the same model due to its successful adaptation to the sentiment classification task.
Criteria for Selection:
Adaptability: GPT-2's ability to fine-tune effectively for specific tasks is demonstrated by its improved performance on sentiment classification.
Performance Gains: Post-fine-tuning, the model’s ability to specialize and perform better in the sentiment analysis domain is a key factor in retaining the model.

2. Task-Specific Considerations for Fine-Tuning

Explanation:

Task: Sentiment classification of movie reviews.
Considerations:
Fine-Tuning Specifics: Fine-tuning adapts GPT-2 to understand and classify sentiment more effectively by learning from IMDB dataset examples.
Specialized Performance: The model now excels in recognizing sentiment nuances and patterns specific to movie reviews, improving upon its general pre-trained capabilities.

3. Data Preparation and Preprocessing Steps

Explanation:

Dataset Preparation: The IMDB dataset is preprocessed specifically for sentiment classification.
Preprocessing Steps:
Tokenization: Applied to convert movie reviews into tokens, fitting the input requirements for GPT-2.
Padding and Truncation: Ensured sequences fit the model’s input size, optimizing training efficiency.
Data Splitting: Split data into training and testing sets to evaluate performance effectively post-fine-tuning.

4. Fine-Tuning Hyperparameters and Optimization Strategies

Explanation:

Hyperparameters:
Learning Rate: Set to 2e-5 for fine-tuning, balancing between convergence speed and stability.
Batch Size: Configured to 8 for managing memory and computational efficiency.
Number of Epochs: Set to 3, ensuring sufficient training while preventing overfitting.
Weight Decay: Applied at 0.01 to regularize the model and reduce overfitting risks.
Optimization Strategies:
Evaluation Strategy: Evaluated model performance at each epoch to monitor and adjust training as needed.
Logging: Used to track progress and adjust parameters based on training dynamics.

5. Evaluation Metrics and Performance Analysis

Explanation:

Metrics:
Loss: The primary metric to evaluate how well the model has learned to classify sentiments. After fine-tuning, loss should decrease, indicating better model performance.
Accuracy: Assessing classification accuracy post-fine-tuning reveals improvements in the model's ability to correctly classify sentiment.
Performance Analysis:
Improved Metrics: Post-fine-tuning results should show lower loss and higher accuracy compared to pre-fine-tuning.
Visual Comparisons: Graphs or charts comparing metrics before and after fine-tuning provide clear insights into performance improvements.

# Summary
## Before Fine-Tuning:

Model: Pre-trained GPT-2
Capabilities: General text generation, zero-shot classification
Performance: Limited accuracy for sentiment classification tasks, generalized results.

## After Fine-Tuning:
Model: Fine-tuned GPT-2 on IMDB dataset
Capabilities: Specialized sentiment classification
Performance: Improved accuracy and relevance, lower loss, and better classification results.
