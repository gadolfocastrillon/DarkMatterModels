import rutinas as r
import time 
import confindence as c

if __name__ == '__main__':
	print("Running de_scan") 
	tO = time.time()
	x,call = r.de_scan()
	de_time = time.time() - tO 
	de_time = de_time/60
	print("Tiempo de ejecución: ", de_time, " minutos")
	print("Finalizado")