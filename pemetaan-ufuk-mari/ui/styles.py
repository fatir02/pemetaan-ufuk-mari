"""
styles.py
Tema UI modern untuk aplikasi instrumen falak pemetaan ufuk mar'i.
Nuansa Dark Slate dengan aksen Crimson/Red dan Teal, kontras tinggi, dan tipografi tajam.
"""

DARK_THEME_QSS = """
/* Global Window & Font */
QWidget {
    background-color: #121417;
    color: #e0e6ed;
    font-family: 'Segoe UI', 'Ubuntu', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
    selection-background-color: #b71c1c;
    selection-color: #ffffff;
}

/* Header Container */
#HeaderFrame {
    background-color: #171a21;
    border-bottom: 2px solid #b71c1c;
    padding: 10px 20px;
}

#AppTitleLabel {
    font-size: 18px;
    font-weight: bold;
    color: #ffffff;
    letter-spacing: 0.5px;
}

#AppSubTitleLabel {
    font-size: 11px;
    color: #90a4ae;
    font-weight: 500;
}

#StatusBadge {
    background-color: #1e232d;
    border: 1px solid #37474f;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
    color: #4fc3f7;
    font-weight: 600;
}

/* Navigasi Utama */
#NavContainer {
    background-color: #14171d;
    border-bottom: 1px solid #262c36;
}

QPushButton.nav-btn {
    background-color: #1a1e27;
    border: none;
    border-bottom: 3px solid transparent;
    color: #b0bec5;
    font-size: 13px;
    font-weight: bold;
    padding: 12px 24px;
    border-radius: 0px;
}

QPushButton.nav-btn:hover {
    background-color: #232936;
    color: #ffffff;
}

QPushButton.nav-btn:checked, QPushButton.nav-btn.active {
    background-color: #222733;
    color: #ff5252;
    border-bottom: 3px solid #ff1744;
}

/* Cards & GroupBox */
QGroupBox {
    background-color: #181c24;
    border: 1px solid #2b3240;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 16px;
    font-weight: bold;
    color: #eceff1;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #ff5252;
    background-color: #181c24;
}

QFrame.card {
    background-color: #181c24;
    border: 1px solid #2b3240;
    border-radius: 6px;
    padding: 10px;
}

/* Form Controls */
QLabel {
    color: #cfd8dc;
}

QLabel.sub-label {
    color: #78909c;
    font-size: 11px;
}

QLabel.field-label {
    font-weight: 600;
    color: #e2e8f0;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
    background-color: #202530;
    border: 1px solid #374151;
    border-radius: 4px;
    padding: 6px 10px;
    color: #ffffff;
    selection-background-color: #c62828;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
    border: 1px solid #ff5252;
    background-color: #242a38;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #1e232d;
    border: 1px solid #374151;
    selection-background-color: #b71c1c;
    color: #ffffff;
}

/* Buttons */
QPushButton {
    background-color: #2a3140;
    border: 1px solid #3d475a;
    border-radius: 4px;
    color: #ffffff;
    font-weight: 600;
    padding: 7px 15px;
}

QPushButton:hover {
    background-color: #384256;
    border-color: #54647f;
}

QPushButton:pressed {
    background-color: #1f242f;
}

QPushButton:disabled {
    background-color: #1a1d24;
    border-color: #2a2e39;
    color: #546070;
}

/* Primary Crimson Button */
QPushButton.btn-primary {
    background-color: #b71c1c;
    border: 1px solid #d32f2f;
    color: #ffffff;
}

QPushButton.btn-primary:hover {
    background-color: #d32f2f;
    border-color: #f44336;
}

QPushButton.btn-primary:pressed {
    background-color: #8b0000;
}

/* Action Teal Button */
QPushButton.btn-teal {
    background-color: #00695c;
    border: 1px solid #00897b;
    color: #ffffff;
}

QPushButton.btn-teal:hover {
    background-color: #00897b;
    border-color: #26a69a;
}

/* Indigo Secondary Button */
QPushButton.btn-indigo {
    background-color: #283593;
    border: 1px solid #3949ab;
    color: #ffffff;
}

QPushButton.btn-indigo:hover {
    background-color: #3949ab;
    border-color: #5c6bc0;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #2d3748;
    height: 6px;
    background: #1a202c;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #ff5252;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #ff1744;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #ffcdd2;
    border-color: #d50000;
}

/* CheckBox */
QCheckBox {
    color: #cfd8dc;
    spacing: 7px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #4a5568;
    border-radius: 3px;
    background: #202530;
}

QCheckBox::indicator:checked {
    background: #d32f2f;
    border-color: #ff5252;
}

/* ScrollBar */
QScrollBar:vertical {
    border: none;
    background: #14171d;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #2f3847;
    border-radius: 4px;
    min-height: 25px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""
