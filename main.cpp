#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <deque>
#include <map>
#include <thread>
#include <future>
#include <boost/compute.hpp>
#include "synthSignal/Signal.hpp"
#include "synthSignal/SineWaveform.hpp"
#include "minusDarwin/Solver.hpp"
#include <Utility.hpp>


namespace bc = boost::compute;

const float signalTime = 1.0f;
const unsigned int sps = 8000;
const float a = 127.0f*sqrt(2);
const float omega = 2.0f*M_PI*60.0f;
const float phi = M_PI;
const float omegaMin = 2.0f*M_PI*60.0f*0.94f;
const float omegaMax = 2.0f*M_PI*60.0f*1.06f;
const float phiMax = 2.0f*M_PI;
const size_t maxGens = 80;
const size_t replications = 100;

struct cmpDevice{
    bool operator()(const bc::device& lhs, const bc::device &rhs) {
        return lhs.id() < rhs.id();
    }
};
const char sinewaveFitSource[] =
        BOOST_COMPUTE_STRINGIZE_SOURCE(
                __kernel void calculateScoresOfPopulation(
                        __global const float *signalData,
                __global const float2 *population,
                __global float *scores,
                const unsigned int populationSize,
                const unsigned int signalDataSize,
                const float sumOfSquares,
                const float omegaMin,
                const float omegaMax,
                const float phiMax,
                const unsigned int sps,
                const float a)
        {
                const uint aId = get_global_id(0);
                const float2 agent = population[aId];
                float error = 0.0f;
                for (size_t p = 0; p < signalDataSize; p++) {
                float t = (float)p/(float)sps;
                float realOmega = omegaMin+agent.x*(omegaMax-omegaMin);
                float realPhi = agent.y*phiMax;
                float estimated =
                a*sin(realOmega*t+realPhi);
                error += (estimated-signalData[p])*(estimated-signalData[p]);
        }
                scores[aId] = error/sumOfSquares;
        });
const std::string outputFilename = "SinewaveFitterTuning.csv";
std::deque<MinusDarwin::SolverParameterSet> *generateConfigurations();
bc::device getFreeDevice(std::map<bc::device,std::future<void>,cmpDevice> &deviceThreadMap);
void runBatchFit(
        bc::device device,
        bc::context *ctx,
        bc::command_queue *queue,
        bc::event *event,
        bc::vector<float> *dSignal,
        std::vector<MinusDarwin::RunTracer> &traces,
        bc::kernel *kernel,
        const MinusDarwin::SolverParameterSet config,
        const size_t replications,
        const float &sumOfSquares);
void writeResults(std::ofstream &of,
                  MinusDarwin::SolverParameterSet &config,
                  std::vector<MinusDarwin::RunTracer> &traces);
int main() {
    std::cout << "Initializing test..." << std::endl;
    std::ofstream of(outputFilename.c_str(),std::ofstream::out);
    auto configsDeque = generateConfigurations();
    of << "popSize,mode,modeDepth,coProb,diffFactor,useUniform,gen,mean,sd" << std::endl;
    auto devicesCount = bc::system::device_count();
    auto devices = std::vector<bc::device>();
    for(auto device : bc::system::devices())
        if(device.type() == CL_DEVICE_TYPE_GPU)
            devices.push_back(device);
    std::map<bc::device,bc::context *,cmpDevice> deviceContextMap;
    std::map<bc::device,bc::command_queue *,cmpDevice> deviceQueueMap;
    std::map<bc::device,bc::event *,cmpDevice> deviceEventMap;
    std::map<bc::device,bc::vector<float> *,cmpDevice> deviceSignalMap;
    std::map<bc::device,std::vector<MinusDarwin::RunTracer> ,cmpDevice> deviceTraceMap;
    std::map<bc::device,bc::kernel *,cmpDevice> deviceKernelMap;
    std::map<bc::device,std::future<void>,cmpDevice> deviceThreadMap;
    std::map<bc::device,MinusDarwin::SolverParameterSet,cmpDevice> deviceConfigMap;

    //Create synthetic signal
    SynthSignal::Signal signalModel;
    SynthSignal::Interpolation interpolation({},SynthSignal::InterpolationType::LINEAR);
    interpolation.addPoint(0.0f,1.0f);
    interpolation.addPoint(signalTime,1.0f);
    SynthSignal::Interpolation frequencyVariation({},SynthSignal::InterpolationType::LINEAR);
    frequencyVariation.addPoint(0.0f,1.0f);
    frequencyVariation.addPoint(signalTime,1.0f);
    auto wf = new SynthSignal::SineWaveform(
            a,
            omega,
            phi,
            frequencyVariation
    );
    signalModel.addEvent(wf,interpolation);
    auto signal = signalModel.gen(signalTime,sps);

    const float sumOfSquares =
            std::accumulate(signal->begin(),signal->end(),0.0f,
                            [](float accum,float val) -> float {
                                accum = accum+val*val;
                            });
    //Copy signal to every device and fill map
    for(auto device : devices) {
        auto ctx = new bc::context(device);
        auto queue = new bc::command_queue(*ctx,device);
        auto event = new bc::event();
        auto traces = std::vector<MinusDarwin::RunTracer>();
        auto dSignal = new bc::vector<float>(
                signal->begin(),signal->end(),*queue);
        auto program = bc::program::create_with_source(sinewaveFitSource, *ctx);
        // compile the program
        try {
            program.build();
        } catch(bc::opencl_error &e) {
            std::cout << program.build_log() << std::endl;
        }
        auto kernel = new bc::kernel(program,"calculateScoresOfPopulation");
        kernel->set_arg(0,*dSignal);
        //kernel->set_arg(1,*dX);
        //kernel->set_arg(2,*dScores);
        //kernel->set_arg(3,(unsigned int)dX->size());
        kernel->set_arg(4,(unsigned int)dSignal->size());
        kernel->set_arg(5,sumOfSquares);
        kernel->set_arg(6,omegaMin);
        kernel->set_arg(7,omegaMax);
        kernel->set_arg(8,phiMax);
        kernel->set_arg(9,(unsigned int)sps);
        kernel->set_arg(10,a);


        deviceContextMap[device] = ctx;
        deviceQueueMap[device] = queue;
        deviceEventMap[device] = event;
        deviceTraceMap[device] = traces;
        deviceSignalMap[device] = dSignal;
        deviceKernelMap[device] = kernel;
    }
    for(auto &device : devices) {
        auto config = configsDeque->front();
        configsDeque->pop_front();
        deviceConfigMap[device] = config;
        deviceThreadMap[device] = std::async(std::launch::async,runBatchFit,
                                                device,
                                                deviceContextMap[device],
                                                deviceQueueMap[device],
                                                deviceEventMap[device],
                                                deviceSignalMap[device],
                                                std::ref(deviceTraceMap[device]),
                                                deviceKernelMap[device],
                                                config,
                                                replications,
                                                std::ref(sumOfSquares));
    }
    while(configsDeque->size()>0) {
        //Wait for an idle device
        auto device = getFreeDevice(deviceThreadMap);
        //Export results to csv
        writeResults(of,deviceConfigMap[device],deviceTraceMap[device]);

        //Get a configuration
        auto config = configsDeque->front();
        configsDeque->pop_front();
        deviceConfigMap[device] = config;
        //Add to device queue the run
        //When the task ends the device must be
        //added to the idleDevices deque
        deviceThreadMap[device] = std::async(std::launch::async,runBatchFit,
                                     device,
                                     deviceContextMap[device],
                                     deviceQueueMap[device],
                                     deviceEventMap[device],
                                     deviceSignalMap[device],
                                     std::ref(deviceTraceMap[device]),
                                     deviceKernelMap[device],
                                     config,
                                     replications,
                                     std::ref(sumOfSquares));
    }
    //Wait for all to finish
    for(auto &pair : deviceThreadMap) {
        pair.second.wait();
    }
    of.close();
    delete(wf);
    delete(configsDeque);
    return 0;
}
void writeResults(std::ofstream &of,
                  MinusDarwin::SolverParameterSet &config,
                  std::vector<MinusDarwin::RunTracer> &traces) {
    auto replications = traces.size();
    std::vector<std::vector<float> > bestScoresByGeneration(maxGens+1,
    std::vector<float>(replications,0.0f));
    std::vector<float> meanScoreByGeneration(maxGens+1);
    std::vector<float> sdScoreByGeneration(maxGens+1);

    for(size_t g = 0;g<maxGens; g++) {
        for(size_t r = 0; r < replications; r++) {
            auto genScores = &traces.at(r).generationsScores.at(g);
            auto bestScoreInGen =
                    *std::min_element(genScores->begin(),genScores->end());
            bestScoresByGeneration.at(g).at(r) = bestScoreInGen;
        }
    }
    for(size_t g=0;g<maxGens;g++) {
        auto bestScores = &bestScoresByGeneration.at(g);
        float accum1 = std::accumulate(bestScores->begin(),bestScores->end(),0.0f);
        float mean = accum1/(float)replications;
        meanScoreByGeneration.at(g) = mean;
        float accum2 =
            std::accumulate(bestScores->begin(),bestScores->end(),0.0f,[&mean](float accum,float val) {
                return accum+pow(val-mean,2);
        });
        float sd = sqrt(accum2/(float)replications);
        of << config.popSize <<","<<
            (config.mode==MinusDarwin::CrossoverMode::Best?"Best":"Random") <<","
           << config.modeDepth <<","<<config.coProb <<","<<config.diffFactor <<","
           <<config.useUniformFactor<<","<<g<<","
           << mean << "," << sd << std::endl;
    }
}
std::deque<MinusDarwin::SolverParameterSet> *generateConfigurations() {

    const std::vector<size_t> popSizeVector = { 25, 50, 100, 200, 400 };
    const std::vector<MinusDarwin::CrossoverMode> modeVector = {
            MinusDarwin::CrossoverMode::Best,
            MinusDarwin::CrossoverMode::Random
    };
    const std::vector<size_t> modeDepthVector = { 1, 2, 3 };
    const std::vector<float> coProbVector = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
    const std::vector<float> diffFactorVector = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
    const std::vector<bool> useUniformVector = { false, true };
    auto configsNumber =
            popSizeVector.size()*
            modeVector.size()*
            modeDepthVector.size()*
            coProbVector.size()*
            diffFactorVector.size()*
            useUniformVector.size();
    auto configsDeque =
            new std::deque<MinusDarwin::SolverParameterSet>();
    for(auto popSize : popSizeVector)
        for(auto mode : modeVector)
            for(auto modeDepth : modeDepthVector)
                for(auto coProb : coProbVector)
                    for(auto diffFactor : diffFactorVector)
                        for(auto useUniform : useUniformVector)
                        {
                            MinusDarwin::SolverParameterSet config = {
                                    2,
                                    popSize,
                                    maxGens,
                                    MinusDarwin::GoalFunction::MaxGenerations,
                                    mode,
                                    modeDepth,
                                    0.005f,
                                    coProb,
                                    diffFactor,
                                    useUniform
                            };
                            configsDeque->push_back(config);
                        }
    return configsDeque;
}
bc::device getFreeDevice(std::map<bc::device, std::future<void>,cmpDevice > &deviceThreadMap) {
    auto device = bc::device();
    while(true)
        for(auto &pair : deviceThreadMap)
            if(pair.second.wait_for(std::chrono::seconds(0)) ==
                    std::future_status::ready) {
                return(pair.first);
            }
}

void runBatchFit(
        bc::device device,
        bc::context *ctx,
        bc::command_queue *queue,
        bc::event *event,
        bc::vector<float> *dSignal,
        std::vector<MinusDarwin::RunTracer> &traces,
        bc::kernel *kernel,
        const MinusDarwin::SolverParameterSet config,
        const size_t replications,
        const float &sumOfSquares) {
    traces.clear();
    auto dX = new bc::vector<bc::float2_>(config.popSize,*ctx);
    auto dScores = new bc::vector<float>(config.popSize,*ctx);

    kernel->set_arg(1,*dX);
    kernel->set_arg(2,*dScores);
    kernel->set_arg(3,(unsigned int)dX->size());
    // compile the program
    auto evaluatePopulationLambda =
            [queue, kernel, dSignal, dScores, dX, sumOfSquares](
                    MinusDarwin::Scores &scores,
                    const MinusDarwin::Population &population) {
                //Population to Device
                std::vector<bc::float2_> hPopulation(population.size());
                std::transform(population.begin(),population.end(),
                               hPopulation.begin(),[](const MinusDarwin::Agent &a){
                            bc::float2_ b;
                            b[0] = a.at(0);
                            b[1] = a.at(1);
                            return b;
                        });
                bc::copy(hPopulation.begin(),hPopulation.end(),dX->begin(),*queue);
                //Once population has been copied to the device
                //a parallel calculation of scores must be done
                queue->enqueue_1d_range_kernel(*kernel,0,hPopulation.size(),0);
                bc::copy(dScores->begin(),dScores->end(),scores.begin(),*queue);
            };
    for(size_t replication = 0; replication < replications; replication++) {
        MinusDarwin::Solver solver(config,evaluatePopulationLambda);
        solver.run(false);
        traces.push_back(solver.tracer);
    }
    free(dX);
    free(dScores);
}