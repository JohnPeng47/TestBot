from .langs import SupportedLangs

class Interpreter:
    def __init__(self, )


class Linter:
    def __init__(self, lang: SupportedLangs):
        self._lang = lang
        self._interp = self._find_lang_interperter()     
        self._linter = self._check_installed_linter()       
        
    def _check_installed_linter(self):
        if self._lang == SupportedLangs.Python:
            return True
        return False
    
    def _find_lang_interperter(self, lang: SupportedLangs) -> Interpreter:
        if self._interp == SupportedLangs.Python:
            return "python"
        return None
    
    def to_linted_code(self, lines) -> str:
        """
        Lint generated code file
        """
        self.i = 0 if getattr(self, "i", None) is None else self.i + 1

        black_cmd_str = f"{self._interp} -m black "
        tmp_file = f"/tmp/test{str(self.i)}.py"

        with open(tmp_file, mode="w+t", encoding="utf-8") as temp_file:
            temp_file.write("\n".join(lines))
            temp_file.flush()

            process = subprocess.Popen(
                black_cmd_str + tmp_file,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()
            if stderr:
                stderr = stderr.decode("utf-8")
                if "error:" in stderr:
                    raise LintException(f"Error while linting: {stderr}")

            with open(tmp_file, "r") as temp_file:
                linted_code = temp_file.read()

        return linted_code
