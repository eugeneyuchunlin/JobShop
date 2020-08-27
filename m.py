
class Job:
    start_time = 0
    end_time = 0
    process_time = {}
    輪盤法 = {} 
    machine_number = 0
    probability = 0
    def __init__(self, process_time):
        # self.process_time = process_time
        self.process_time = {
                1 : 80,
                2 : 60,
                3 : 50
        }
        
        keys = self.process_time.keys()
        machine_numbers = len(keys)

        split = 1 / machine_numbers  # 0.3333
        for i in range(len(keys)):
            self.輪盤法[i + 1] = split * (i + 1) # 1 : 0.333, 2 : 0.666 , 3 : 0.999
        


    def set_probability(self, pro):

        self.probability = pro
        
        for i in range(len(self.process_time.keys())):
            if(pro < self.輪盤法[i + 1]):
                self.machine_number = i


    

class Machine:
    jobs = []
    max_time = 0
    number = 0
    total_time = 0
    def __init__(self):
        self.max_time = 0
        self.total_time = 0
        pass

    def add_job(self):
        pass


    def clear(self):
        pass

    def get_total_time(self):

        return total_time

class Chromosome:
    probabilities = []
    number = 0
    size = 0

    def __init__(self, number, size):
        self.number = number
        self.size = size
        for i in range(2 * size):
            self.probabilities.append(self.random())

    def random(self):
        return 0.67

    def get_probability(self, job_number):
        return probabilities[job_number] 

if __name__ == '__main__':
    job = Job(1)
    ch = Chromosome(10, 10)
    job.set_probability(ch.get_probability(1))
