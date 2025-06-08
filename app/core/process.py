import multiprocessing
import logging
from typing import *
from core.entities_llm import EngineComunication
from core.rkllm import RKLLM, global_state


class RKLLM_Engine:

    def __init__(self, engine_params: Dict[str,Any],
                 cmd_queue: 'multiprocessing.Queue[EngineComunication]',
                 control_queue: multiprocessing.Queue):
        self.cmd_queue = cmd_queue
        self.control_queue = control_queue
        self.engine_params=engine_params
        pass

    def worker_func(self):
        # Esegue eventuali inizializzazioni della libreria se necessario
        self.engine = RKLLM(**self.engine_params)
        logging.info("Loaded")
        while True:
            cmd = self.cmd_queue.get()  # Bloccante finché non arriva un comando
            if cmd == "STOP":
                # Segnale per terminare il processo
                break

            try:
                match cmd.function_name:
                    case "run":
                        self.engine.run(**cmd.params)
                    case "abort_job":
                        _res = self.engine.abort_job(**cmd.params)
                        self.control_queue.put({
                            "fun": cmd.function_name,
                            "res": _res
                        })
                    case "is_running":
                        _res = self.engine.is_running(**cmd.params)
                        self.control_queue.put({
                            "fun": cmd.function_name,
                            "res": _res
                        })

            except Exception as e:
                # Gestisce eventuali eccezioni e le invia alla coda dei risultati
                logging.error(f"Engine: {e}")
                self.control_queue.put({
                    "fun": cmd.function_name,
                    "ex": str(e)
                })

        self.engine.release()
