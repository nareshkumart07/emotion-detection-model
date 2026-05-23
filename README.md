<!-- doc:hero -->
<div align="center">

<h1>Emotion Detection Model</h1>

<p><strong>Emotion Detection Model</strong> • <em>Project Documentation</em></p>

<p>
  <code>DREAMER</code>
  &nbsp;•&nbsp;
  <code>EEG</code>
  &nbsp;•&nbsp;
  <code>BiLSTM</code>
  &nbsp;•&nbsp;
  <code>EEGNet</code>
</p>

</div>

---

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Detection Model</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 24px;
            margin-bottom: 12px;
        }
        h1 { font-size: 2.5em; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { font-size: 2em; margin-top: 30px; }
        h3 { font-size: 1.5em; }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
            background-color: white;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        pre {
            background-color: #f4f4f4;
            border-left: 4px solid #3498db;
            padding: 12px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        section {
            background-color: white;
            padding: 20px;
            margin: 16px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        ul, ol {
            margin: 12px 0;
            padding-left: 30px;
        }
        li {
            margin: 8px 0;
        }
        hr {
            border: none;
            border-top: 2px solid #ddd;
            margin: 24px 0;
        }
        strong {
            color: #2c3e50;
            font-weight: 600;
        }
        em {
            font-style: italic;
            color: #555;
        }
    </style>
</head>
<body>
<h1>Emotion Detection Model</h1>

EEG emotion classification project with:
<ul>
<ol>
<li>4-class inference (<strong>INLINE<em>CODE</em>0</strong>, <strong>INLINE<em>CODE</em>1</strong>, <strong>INLINE<em>CODE</em>2</strong>, <strong>INLINE<em>CODE</em>3</strong>)</li>
<li>binary EEGNet training (<strong>INLINE<em>CODE</em>4</strong>, <strong>INLINE<em>CODE</em>5</strong>)</li>
<li>Streamlit dashboard demo</li>
<li>FastAPI inference API</li>
</ol>
</ul>

<h2>Project Structure</h2>

<strong>CODE<em>BLOCK</em>0</strong>

<h2>For Normal User</h2>

<p>Go to the project folder first:</p>

<strong>CODE<em>BLOCK</em>1</strong>

<p>Install dependencies:</p>

<strong>CODE<em>BLOCK</em>2</strong>

<p>Train model (requires <strong>INLINE<em>CODE</em>6</strong> in the project root or <strong>INLINE<em>CODE</em>7</strong>):</p>

<strong>CODE<em>BLOCK</em>3</strong>

<p>Run Streamlit dashboard with the bundled pretrained models:</p>

<strong>CODE<em>BLOCK</em>4</strong>

<p>Run API server:</p>

<strong>CODE<em>BLOCK</em>5</strong>

<p>Evaluate the valence + arousal model pair:</p>

<strong>CODE<em>BLOCK</em>6</strong>

<p>Evaluate with trial-level aggregation (<strong>INLINE<em>CODE</em>8</strong> or <strong>INLINE<em>CODE</em>9</strong>):</p>

<strong>CODE<em>BLOCK</em>7</strong>

<p>Evaluate fold-ensemble performance (recommended for stronger results):</p>

<strong>CODE<em>BLOCK</em>8</strong>

<p>Run multi-configuration search:</p>

<strong>CODE<em>BLOCK</em>9</strong>

<h2>Model Performance Summary</h2>

<h3>4-Class Quadrant Classification (BiLSTM)</h3>
<ul>
<li><strong>Validation accuracy: ~34.6%</strong></li>
<li>Balanced accuracy: 29.7%</li>
<li>Train accuracy: 41.96%</li>
<li><em>Note: Weak performance reflects task difficulty; marginally above majority-class baseline.</em></li>
</ul>

<h3>Binary Classification (EEGNet)</h3>
Due to the challenge of 4-class classification, we also trained separate binary models for <strong>valence</strong> and <strong>arousal</strong> emotions.

Current seed-999 window-level results (<strong>INLINE<em>CODE</em>10</strong>):
<ul>
<li><strong>Valence:</strong> <strong>68.61%</strong> accuracy (balanced accuracy 53.38%, macro F1 0.5345)</li>
<li><strong>Arousal:</strong> <strong>57.17%</strong> accuracy (balanced accuracy 58.38%, macro F1 0.5534)</li>
</ul>

With trial-level aggregation (<strong>INLINE<em>CODE</em>11</strong>):
<ul>
<li><strong>Valence:</strong> up to <strong>75.00%</strong> (<strong>INLINE<em>CODE</em>12</strong>)</li>
<li><strong>Arousal:</strong> <strong>61.96%</strong></li>
<li><strong>Quadrant from binary:</strong> <strong>44.57%</strong></li>
</ul>

<h3>Recent Seed Sweep (Cross-Trial, 1 Fold)</h3>
Latest local tuning results from <strong>INLINE<em>CODE</em>13</strong> with <strong>INLINE<em>CODE</em>14</strong>:

<table>
<thead>
<tr><th>Seed</th><th>Task</th><th>Accuracy</th><th>Balanced Accuracy</th><th>Macro F1</th></tr>
</thead>
<tbody>
<tr><td>42</td><td>Valence</td><td>64.05%</td><td>49.42%</td><td>0.4901</td></tr>
<tr><td>42</td><td>Arousal</td><td>46.56%</td><td>53.62%</td><td>0.4649</td></tr>
<tr><td>999</td><td>Valence</td><td>68.61%</td><td>53.38%</td><td>0.5345</td></tr>
<tr><td>999</td><td>Arousal</td><td>57.17%</td><td>58.38%</td><td>0.5534</td></tr>
</tbody>
</table>
Best run so far: <strong>seed 999</strong> for both valence and arousal.

<p>Reproduce these commands:</p>

<strong>CODE<em>BLOCK</em>10</strong>

<h3>Reproducible Seed-999 Workflow</h3>

Use this exact flow to reproduce the current best local run:
<li>Train:</li>

<strong>CODE<em>BLOCK</em>11</strong>
<li>Open reports:</li>

<strong>CODE<em>BLOCK</em>12</strong>
<li>Expected metrics (single-fold <strong>INLINE<em>CODE</em>15</strong>):</li>
<ul>
<li>Valence: accuracy <strong>INLINE<em>CODE</em>16</strong>, balanced accuracy <strong>INLINE<em>CODE</em>17</strong>, macro F1 <strong>INLINE<em>CODE</em>18</strong></li>
<li>Arousal: accuracy <strong>INLINE<em>CODE</em>19</strong>, balanced accuracy <strong>INLINE<em>CODE</em>20</strong>, macro F1 <strong>INLINE<em>CODE</em>21</strong></li>
</ul>
<li>Optional trial-level paired evaluation:</li>

<strong>CODE<em>BLOCK</em>13</strong>
<li>Expected trial-level results (<strong>INLINE<em>CODE</em>22</strong>):</li>
<ul>
<li>Valence: accuracy <strong>INLINE<em>CODE</em>23</strong></li>
<li>Arousal: accuracy <strong>INLINE<em>CODE</em>24</strong></li>
<li>Quadrant from binary: accuracy <strong>INLINE<em>CODE</em>25</strong></li>
</ul>

<h2>Important Notes</h2>
<ul>
<li>Manual preprocessing is not required. Training handles it automatically.</li>
<li><strong>INLINE<em>CODE</em>26</strong> is needed for training and evaluation, but not for opening the Streamlit demo or API with the included model files.</li>
<li>If GPU is available, <strong>INLINE<em>CODE</em>27</strong> uses GPU; otherwise CPU is used.</li>
<li>Advanced commands are documented in <a href="docs/TRAIN<em>AND</em>STREAMLIT.md">docs/TRAIN<em>AND</em>STREAMLIT.md</a>.</li>
<li>Python file roles are documented in <a href="docs/FILE</em>GUIDE.md">docs/FILE<em>GUIDE.md</a>.</li>
<li>Full technical details are documented in <a href="docs/IMPLEMENTATION</em>DETAILS.md">docs/IMPLEMENTATION<em>DETAILS.md</a>.</li>
<li>Viva prep Q&A is available at <a href="docs/VIVA</em>QUESTIONS.pdf">docs/VIVA<em>QUESTIONS.pdf</a>.</li>
</ul>

<h2>Submission Support Docs</h2>
<ul>
<li>Project title, abstract, objectives: <a href="docs/ABSTRACT<em>AND</em>OBJECTIVES.md">docs/ABSTRACT<em>AND</em>OBJECTIVES.md</a></li>
<li>Viva Q&A sheet: <a href="docs/VIVA</em>QUESTIONS.md">docs/VIVA<em>QUESTIONS.md</a></li>
<li>Presentation script: <a href="docs/PRESENTATION</em>SCRIPT.md">docs/PRESENTATION<em>SCRIPT.md</a></li>
<li>Full report with limitations and future work: <a href="docs/PROJECT</em>REPORT.md">docs/PROJECT<em>REPORT.md</a></li>
<li>Accuracy optimization workflow: <a href="docs/ACCURACY</em>OPTIMIZATION.md">docs/ACCURACY<em>OPTIMIZATION.md</a></li>
</ul>
</body>
</html>
