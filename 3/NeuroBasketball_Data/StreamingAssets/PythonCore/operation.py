from quality_monitor import QualityMonitor
import os

def init(root):
    return QualityMonitor(os.path.join(root, 'scaler.pickle'), os.path.join(root, 'classifier.pickle'),os.path.join(root, 'labels.json'), os.path.join(root, 'extraction_settings.json'))

def load(qm, values):
    qm.update_buffer(values)
    qm.predict()
    return qm.get_colors()
