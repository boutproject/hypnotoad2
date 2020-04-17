"""
GUI for Hypnotoad2 using Qt

"""

import numbers
import os
import yaml

from Qt.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QMessageBox,
)

from .hypnotoad2_mainWindow import Ui_Hypnotoad2
from .matplotlib_widget import MatplotlibWidget
from ..cases import tokamak
from ..core.mesh import BoutMesh


colours = {
    "red": "#aa0000",
}


def convert_python_type_to_qwidget(value):
    """
    Convert a python type into the appropriate Qt widget
    """
    if isinstance(value, bool):
        return QCheckBox
    if isinstance(value, numbers.Integral):
        return QSpinBox
    if isinstance(value, numbers.Real):
        return QDoubleSpinBox
    if isinstance(value, str):
        return QLineEdit
    return QLineEdit


class HypnotoadGui(QMainWindow, Ui_Hypnotoad2):
    """A graphical interface for Hypnotoad2

    """

    def __init__(self):
        super().__init__(None)
        self.setupUi(self)

        self.plot_widget = MatplotlibWidget(self.plottingArea)

        self.geqdsk_file_browse_button.clicked.connect(self.select_geqdsk_file)
        self.geqdsk_file_browse_button.setToolTip(
            self.select_geqdsk_file.__doc__.strip()
        )
        self.geqdsk_file_line_edit.editingFinished.connect(self.read_geqdsk)
        self.options_file_browse_button.clicked.connect(self.select_options_file)
        self.options_file_browse_button.setToolTip(
            self.select_options_file.__doc__.strip()
        )
        self.options_file_line_edit.editingFinished.connect(self.read_options)

        self.run_button.clicked.connect(self.run)
        self.run_button.setToolTip(self.run.__doc__.strip())

        self.write_grid_button.clicked.connect(self.write_grid)
        self.write_grid_button.setToolTip(self.write_grid.__doc__)
        self.write_grid_button.setEnabled(False)

        self.options = dict(tokamak.TokamakEquilibrium.default_options.items())

        for key, value in sorted(self.options.items()):
            self.add_options_widget(key, value)

    def add_options_widget(self, key, value):
        """Take a key, value pair and add a row with the appropriate widget
        to the options form

        """
        widget_type = convert_python_type_to_qwidget(value)

        widget = widget_type()

        if isinstance(value, bool):
            widget.setChecked(value)
            widget.stateChanged.connect(lambda state: self.options.update(key=value))
        elif isinstance(value, numbers.Integral):
            widget.setMaximum(100000)
            widget.setValue(value)
            widget.valueChanged.connect(lambda value: self.options.update(key=value))
        elif isinstance(value, numbers.Real):
            widget.setDecimals(8)
            widget.setRange(-1e300, 1e300)
            widget.setValue(value)
            widget.valueChanged.connect(lambda value: self.options.update(key=value))
        elif isinstance(value, str):
            widget.setText(value)
            widget.textChanged.connect(lambda text: self.options.update(key=text))
        else:
            widget.textChanged.connect(lambda text: self.options.update(key=text))

        widget.setObjectName(key)
        self.options_form_layout.addRow(key, widget)
        return widget

    def update_options_form(self):
        """Update the widget values in the options form, based on the current
        values in the options dict

        """
        for key, value in sorted(self.options.items()):
            widget_type = convert_python_type_to_qwidget(value)
            widget = self.findChild(widget_type, key)

            # If we didn't already know the type, then it would be a
            # QLineEdit instead of a more specific widget
            if widget is None:
                widget = self.findChild(QLineEdit, key)
                if widget is not None:
                    self.options_form_layout.removeRow(widget)
                widget = self.add_options_widget(key, value)

            if isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, QSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(value)
            else:
                raise RuntimeError(
                    f"Unknown widget when trying to update options ({type(widget)})"
                )

    def select_options_file(self):
        """Choose a Hypnotoad2 options file to load

        """

        filename, _ = QFileDialog.getOpenFileName(
            self, "Open options file", ".", filter="YAML file (*.yml *.yaml)"
        )

        if (filename is None) or (filename == ""):
            return  # Cancelled
        if not os.path.exists(filename):
            self.write("Could not find " + filename)
            return

        self.options_file_line_edit.setText(filename)
        self.read_options()

    def read_options(self):
        """Read the options file

        """

        self.statusbar.showMessage("Reading options", 2000)
        options_filename = self.options_file_line_edit.text()

        if options_filename:
            with open(options_filename, "r") as f:
                self.options.update(yaml.safe_load(f))

        self.update_options_form()

    def select_geqdsk_file(self):
        """Choose a "geqdsk" equilibrium file to open

        """

        filename, _ = QFileDialog.getOpenFileName(self, "Open geqdsk file", ".")

        if (filename is None) or (filename == ""):
            return  # Cancelled
        if not os.path.exists(filename):
            self.write("Could not find " + filename)
            self.geqdsk_file_line_edit.setStyleSheet(
                f"QLineEdit {{ background-color: {colours['red']} }}"
            )
            return

        self.geqdsk_file_line_edit.setText(filename)
        self.geqdsk_file_line_edit.setStyleSheet("")

        self.read_geqdsk()

    def read_geqdsk(self):
        """Read the equilibrium file

        """

        self.statusbar.showMessage("Reading geqdsk", 2000)
        geqdsk_filename = self.geqdsk_file_line_edit.text()

        if not os.path.exists(geqdsk_filename):
            self.geqdsk_file_line_edit.setStyleSheet(
                f"QLineEdit {{ background-color : {colours['red']} }}"
            )
            self.statusbar.showMessage(
                f"Could not find equilibrium file '{geqdsk_filename}'"
            )
            return

        with open(geqdsk_filename, "rt") as fh:
            self.eq = tokamak.read_geqdsk(fh, options=self.options)

        self.eq.plotPotential(ncontours=40, axis=self.plot_widget.axes)
        for region in self.eq.regions.values():
            self.plot_widget.axes.plot(
                [p.R for p in region.points], [p.Z for p in region.points], "-o"
            )

        self.plot_widget.axes.plot(*self.eq.x_points[0], "rx")
        self.plot_widget.canvas.draw()

    def run(self):
        """Run Hypnotoad2 and generate the grid

        """

        if not hasattr(self, "eq"):
            self.statusbar.showMessage("Missing equilibrium file!")
            self.geqdsk_file_line_edit.setStyleSheet(
                f"QLineEdit {{ background-color: {colours['red']} }}"
            )
            return

        self.statusbar.showMessage("Running...")
        self.mesh = BoutMesh(self.eq)
        self.mesh.geometry()
        self.statusbar.showMessage("Done!", 2000)

        self.plot_widget._clean_axes()
        self.eq.plotPotential(ncontours=40, axis=self.plot_widget.axes)
        self.mesh.plotPoints(
            xlow=self.options.get("plot_xlow", True),
            ylow=self.options.get("plot_ylow", True),
            corners=self.options.get("plot_corners", True),
            ax=self.plot_widget.axes,
        )
        self.plot_widget.canvas.draw()

        self.write_grid_button.setEnabled(True)

    def write_grid(self):
        """Write generated mesh to file

        """

        if not hasattr(self, "mesh"):
            message_box = QMessageBox()
            message_box.setText("No mesh found!")
            message_box.exec_()

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save grid to file",
            self.options.get("grid_file", "bout.grd.nc"),
            filter="NetCDF (*nc)",
        )

        self.mesh.writeGridfile(filename)
