"""
GUI for Hypnotoad using Qt

"""

import ast
import copy
import options
import os
import yaml

from Qt.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QCompleter,
    QTableWidgetItem,
    QHeaderView,
    QErrorMessage,
    QDialog,
)
from Qt.QtCore import Qt

from .hypnotoad_mainWindow import Ui_Hypnotoad
from .hypnotoad_preferences import Ui_Preferences
from .matplotlib_widget import MatplotlibWidget
from ..cases import tokamak
from ..core.mesh import BoutMesh
from ..core.equilibrium import SolutionError
from ..__init__ import __version__


COLOURS = {
    "red": "#aa0000",
}


def _table_item_edit_display(item):
    """Hide the "(default)" marker on table items in the options form

    """
    default_marker = " (default)"
    if item.text().endswith(default_marker):
        item.setText(item.text()[: -len(default_marker)])


class HypnotoadGui(QMainWindow, Ui_Hypnotoad):
    """A graphical interface for Hypnotoad

    """

    gui_options = options.Options(
        grid_file="bout.grd.nc",
        plot_xlow=True,
        plot_ylow=True,
        plot_corners=True,
        save_full_yaml=False,
    )

    def __init__(self):
        super().__init__(None)
        self.setupUi(self)

        try:
            self.menu_File.setToolTipsVisible(True)
            self.menu_Mesh.setToolTipsVisible(True)
            self.menu_Help.setToolTipsVisible(True)
        except AttributeError:
            pass

        self.plot_widget = MatplotlibWidget(self.plottingArea)

        def set_triggered(widget, function):
            widget.triggered.connect(function)
            if hasattr(function, "__doc__"):
                widget.setToolTip(function.__doc__.strip())

        def set_clicked(widget, function):
            widget.clicked.connect(function)
            if hasattr(function, "__doc__"):
                widget.setToolTip(function.__doc__.strip())

        set_clicked(self.geqdsk_file_browse_button, self.select_geqdsk_file)
        self.geqdsk_file_line_edit.editingFinished.connect(self.read_geqdsk)

        set_clicked(self.options_file_browse_button, self.select_options_file)
        self.options_file_line_edit.editingFinished.connect(self.read_options)

        set_clicked(self.run_button, self.run)
        set_triggered(self.action_Run, self.run)

        set_clicked(self.write_grid_button, self.write_grid)
        set_triggered(self.action_Write_grid, self.write_grid)
        self.write_grid_button.setEnabled(False)

        set_triggered(self.action_Revert, self.revert_options)
        set_triggered(self.action_Save, self.save_options)
        set_triggered(self.action_Save_as, self.save_options_as)
        set_triggered(self.action_New, self.new_options)
        set_triggered(self.action_Open, self.select_options_file)
        set_triggered(self.action_About, self.help_about)
        set_triggered(self.action_Preferences, self.open_preferences)

        self.action_Quit.triggered.connect(self.close)

        self.options = {}
        self.gui_options = HypnotoadGui.gui_options.push({})
        self.filename = "Untitled.yml"

        self.search_bar.setPlaceholderText("Search options...")
        self.search_bar.textChanged.connect(self.search_options_form)
        self.search_bar.setToolTip(self.search_options_form.__doc__.strip())
        self.search_bar_completer = QCompleter(
            tokamak.TokamakEquilibrium.default_options.keys()
        )
        self.search_bar_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.search_bar.setCompleter(self.search_bar_completer)

        self.options_form.cellChanged.connect(self.options_form_changed)
        self.options_form.itemDoubleClicked.connect(_table_item_edit_display)
        self.update_options_form()

    def help_about(self):
        """About Hypnotoad

        """

        about_text = __doc__.strip()
        about_text += f"\nVersion : {__version__}"

        about_box = QMessageBox(self)
        about_box.setText(about_text)
        about_box.exec_()

    def open_preferences(self):
        """GUI preferences and settings

        """
        preferences_window = Preferences(self)
        preferences_window.exec_()

    def revert_options(self):
        """Revert the current options to the loaded file, or defaults if no
        file loaded

        """

        self.statusbar.showMessage("Reverting options", 2000)
        self.options = {}
        if hasattr(self, "eq"):
            self.eq.updateOptions()

        options_filename = self.options_file_line_edit.text()

        if options_filename:
            self.read_options()
        else:
            self.options_form.setRowCount(0)
            self.update_options_form()

    def new_options(self):
        """New set of options

        """

        self.options = {}
        if hasattr(self, "eq"):
            self.eq.updateOptions()
        self.options_form.setRowCount(0)
        self.update_options_form()

    def save_options(self):
        """Save options to file

        """

        self.statusbar.showMessage("Saving...", 2000)

        if not self.filename or self.filename == "Untitled.yml":
            self.save_options_as()

        self.options_file_line_edit.setText(self.filename)

        options_to_save = self.options
        if self.gui_options["save_full_yaml"]:
            if hasattr(self, "eq"):
                options_ = self.eq.user_options
            else:
                options_ = tokamak.TokamakEquilibrium.default_options.push(self.options)
            # This converts any numpy types to native Python using the tolist()
            # method of any numpy objects/types. Note this does return a scalar
            # and not a list for values that aren't arrays. Also remove any
            # private/magic keys
            options_to_save = {
                key: getattr(value, "tolist", lambda: value)()
                for key, value in dict(options_).items()
                if not key.startswith("_")
            }

        with open(self.filename, "w") as f:
            yaml.dump(options_to_save, f)

    def save_options_as(self):
        """Save options to file with new filename

        """

        self.filename, _ = QFileDialog.getSaveFileName(
            self, "Save grid to file", self.filename, filter="YAML file (*yml *yaml)",
        )

        self.save_options()

    def update_options_form(self):
        """Update the widget values in the options form, based on the current
        values in the options object

        """

        filtered_options = copy.deepcopy(self.options)
        filtered_defaults = dict(tokamak.TokamakEquilibrium.default_options)
        # Skip options handled specially elsewhere
        del filtered_defaults["_magic"]

        self.options_form.setSortingEnabled(False)
        self.options_form.cellChanged.disconnect(self.options_form_changed)
        self.options_form.setRowCount(len(filtered_defaults))

        for row, (key, value) in enumerate(sorted(filtered_defaults.items())):
            item = QTableWidgetItem(key)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.options_form.setItem(row, 0, item)
            if key in filtered_options:
                value_to_set = str(filtered_options[key])
            else:
                value_to_set = f"{value} (default)"
            self.options_form.setItem(row, 1, QTableWidgetItem(value_to_set))

        self.options_form.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.options_form.setSortingEnabled(True)
        self.options_form.cellChanged.connect(self.options_form_changed)

    def options_form_changed(self, row, column):
        """Change the options form from the widget table

        """

        item = self.options_form.item(row, column)

        if column == 0:
            # column 0 is not editable, so this should not be possible
            raise ValueError("Not allowed to change option names")
        else:
            key = self.options_form.item(row, 0).text()

            if item.text() == "":
                # Reset to default
                # Might be better to just keep the old value if nothing is passed, but
                # don't know how to get that
                default_value = tokamak.TokamakEquilibrium.default_options[key]
                self.options_form.cellChanged.disconnect(self.options_form_changed)
                self.options_form.setItem(
                    row, 1, QTableWidgetItem(f"{default_value} (default)")
                )
                self.options_form.cellChanged.connect(self.options_form_changed)
                if key in self.options:
                    del self.options[key]
                if hasattr(self, "eq"):
                    # deleting from this object means self.eq uses the value from
                    # TokamakEquilibrium.default_options
                    del self.eq.user_options[key]
                    self.eq.updateOptions()
                return

            self.options[key] = ast.literal_eval(item.text())
            if hasattr(self, "eq"):
                self.eq.user_options.update(**self.options)
                self.eq.updateOptions()

    def search_options_form(self, text):
        """Search for specific options

        """

        for i in range(self.options_form.rowCount()):
            row = self.options_form.item(i, 0)

            matches = text.lower() in row.text().lower()
            self.options_form.setRowHidden(i, not matches)

    def select_options_file(self):
        """Choose a Hypnotoad options file to load

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
        self.filename = filename
        self.read_options()

    def read_options(self):
        """Read the options file

        """

        self.statusbar.showMessage("Reading options", 2000)
        options_filename = self.options_file_line_edit.text()

        if options_filename:
            with open(options_filename, "r") as f:
                self.options = yaml.safe_load(f)
                if hasattr(self, "eq"):
                    self.eq.updateOptions()

        self.options_form.setRowCount(0)
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
                f"QLineEdit {{ background-color: {COLOURS['red']} }}"
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
                f"QLineEdit {{ background-color : {COLOURS['red']} }}"
            )
            self.statusbar.showMessage(
                f"Could not find equilibrium file '{geqdsk_filename}'"
            )
            return

        try:
            with open(geqdsk_filename, "rt") as f:
                self.eq = tokamak.read_geqdsk(f, options=copy.deepcopy(self.options))
        except (ValueError, RuntimeError) as e:
            error_message = QErrorMessage()
            error_message.showMessage(str(e))
            error_message.exec_()
            return

        self.update_options_form()

        # Delete mesh if it exists, since we have a new self.eq object
        if hasattr(self, "mesh"):
            del self.mesh

        self.plot_grid()

    def run(self):
        """Run Hypnotoad and generate the grid

        """

        if not hasattr(self, "eq"):
            self.statusbar.showMessage("Missing equilibrium file!")
            self.geqdsk_file_line_edit.setStyleSheet(
                f"QLineEdit {{ background-color: {COLOURS['red']} }}"
            )
            return

        # Call read_geqdsk to recreate self.eq object in case any settings needed in
        # __init__ have been changed
        self.read_geqdsk()

        self.statusbar.showMessage("Running...")
        try:
            self.mesh = BoutMesh(self.eq)
        except (ValueError, SolutionError) as e:
            error_message = QErrorMessage()
            error_message.showMessage(str(e))
            error_message.exec_()
            return

        self.mesh.calculateRZ()
        self.statusbar.showMessage("Done!", 2000)

        self.plot_grid()

        self.write_grid_button.setEnabled(True)

    def write_grid(self):
        """Write generated mesh to file

        """

        # Create all the geometrical quantities
        self.mesh.geometry()

        if not hasattr(self, "mesh"):
            flags = QMessageBox.StandardButton.Ok
            QMessageBox.critical(
                self, "Error", "Can't write mesh to file; no mesh found!", flags
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save grid to file",
            self.gui_options["grid_file"],
            filter="NetCDF (*nc)",
        )

        self.mesh.writeGridfile(filename)

    def plot_grid(self):
        self.plot_widget.clear()

        if hasattr(self, "eq"):
            self.eq.plotPotential(ncontours=40, axis=self.plot_widget.axes)
            self.eq.plotWall(axis=self.plot_widget.axes)

        if hasattr(self, "mesh"):
            # mesh exists, so plot the grid points
            self.mesh.plotPoints(
                xlow=self.gui_options["plot_xlow"],
                ylow=self.gui_options["plot_ylow"],
                corners=self.gui_options["plot_corners"],
                ax=self.plot_widget.axes,
            )
        elif hasattr(self, "eq"):
            # no mesh, but do have equilibrium, so plot separatrices
            for region in self.eq.regions.values():
                self.plot_widget.axes.plot(
                    [p.R for p in region.points], [p.Z for p in region.points], "-o"
                )
            self.plot_widget.axes.plot(*self.eq.x_points[0], "rx")

        self.plot_widget.canvas.draw()


class Preferences(QDialog, Ui_Preferences):
    """Dialog box for editing Hypnotoad preferences
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)
        self.parent = parent

        self.defaultGridFileNameLineEdit.setText(self.parent.gui_options["grid_file"])
        self.plotXlowCheckBox.setChecked(self.parent.gui_options["plot_xlow"])
        self.plotYlowCheckBox.setChecked(self.parent.gui_options["plot_ylow"])
        self.plotCornersCheckBox.setChecked(self.parent.gui_options["plot_corners"])
        self.saveFullYamlCheckBox.setChecked(self.parent.gui_options["save_full_yaml"])

    def accept(self):

        self.parent.gui_options.set(
            grid_file=self.defaultGridFileNameLineEdit.text(),
            plot_xlow=self.plotXlowCheckBox.isChecked(),
            plot_ylow=self.plotYlowCheckBox.isChecked(),
            plot_corners=self.plotCornersCheckBox.isChecked(),
            save_full_yaml=self.saveFullYamlCheckBox.isChecked(),
        )

        self.parent.plot_grid()

        super().accept()