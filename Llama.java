package com.llama4j;

import java.nio.FloatBuffer;
import java.util.*;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

record Llama(Configuration configuration, Tokenizer tokenizer, Weights weights) {
    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    public static final class Configuration {
        public final int dim;
        public final int hiddenDim;
        public final int numberOfLayers;
        public final int numberOfHeads;
        public final int numberOfKeyValueHeads;
        public final int vocabularySize;
        public final int contextLength;
        public final float rmsNormEps;
        public final float ropeTheta;
        public final int headSize;

        Configuration withContextLength(int newContextLength) {
            if (newContextLength < 0) {
                return this;
            }
            return new Configuration(this.dim, this.hiddenDim, this.numberOfLayers, this.numberOfHeads, this.numberOfKeyValueHeads, this.vocabularySize, newContextLength, this.rmsNormEps, this.ropeTheta);
        }

        public Configuration(int dim, int hiddenDim, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads, int vocabularySize, int contextLength, float rmsNormEps, float ropeTheta) {
            this.dim = dim;
            this.hiddenDim = hiddenDim;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.headSize = dim / numberOfHeads;
        }
    }

    public static final class Weights {
        public final FloatTensor token_embedding_table;
        public final FloatBuffer[] rms_att_weight;
        public final FloatTensor[] wq;
        public final FloatTensor[] wk;
        public final FloatTensor[] wv;
        public final FloatTensor[] wo;
        public final FloatBuffer[] rms_ffn_weight;
        public final FloatTensor[] w1;
        public final FloatTensor[] w2;
        public final FloatTensor[] w3;
        public final FloatBuffer rms_final_weight;
        public final FloatBuffer freq_cis_real;
        public final FloatBuffer freq_cis_imag;
        public final FloatTensor wcls;

        public Weights(FloatTensor token_embedding_table, FloatBuffer[] rms_att_weight, FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo, FloatBuffer[] rms_ffn_weight, FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatBuffer rms_final_weight, FloatBuffer freq_cis_real, FloatBuffer freq_cis_imag, FloatTensor wcls) {
            this.token_embedding_table = token_embedding_table;
            this.rms_att_weight = rms_att_weight;
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;
            this.rms_ffn_weight = rms_ffn_weight;
            this.w1 = w1;
            this.w2 = w2;
            this.w3 = w3;
            this.rms_final_weight = rms_final_weight;
            this.freq_cis_real = freq_cis_real;
            this.freq_cis_imag = freq_cis_imag;
            this.wcls = wcls;
        }
    }

    public static final class State {

        public final int batchsize;
        public final FloatTensor[] x;
        public final FloatTensor[] xb;
        public final FloatTensor[] xb2;
        public final FloatTensor[] hb;
        public final FloatTensor[] hb2;
        public final FloatTensor[] q;
        public final FloatTensor[] k;
        public final FloatTensor[] v;
        public final FloatTensor[] att;
        public final FloatTensor logits;

        public final FloatTensor[] keyCache;
        public final FloatTensor[] valueCache;

        int idxPrevBlock;

        public int latestToken;

        State(Configuration config, int batchsize) {
            this.batchsize = batchsize;
            this.x = allocate(batchsize, config.dim);
            this.xb = allocate(batchsize, config.dim);
            this.xb2 = allocate(batchsize, config.dim);
            this.hb = allocate(batchsize, config.hiddenDim);
            this.hb2 = allocate(batchsize, config.hiddenDim);
            this.q = allocate(batchsize, config.dim);
            this.k = allocate(batchsize, config.dim);
            this.v = allocate(batchsize, config.dim);
            this.att = allocate(batchsize, config.numberOfHeads, config.contextLength);
            idxPrevBlock = -1;

            this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
            int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
            this.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
            this.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
        }
    }

    static FloatTensor[] allocate(int numTokens, int... dims) {
        return IntStream.range(0, numTokens)
                .mapToObj(i -> ArrayFloatTensor.allocate(dims))
                .toArray(FloatTensor[]::new);
    }

    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        final float finalss = ss;
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }

    static FloatTensor forward(Llama model, State state, int[] tokens, int position, boolean computeLogits) {
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads;
        float sqrtHeadSize = (float) Math.sqrt(headSize);
        final int nTokens = tokens.length;

        Parallel.parallelFor(0, nTokens, t ->
            weights.token_embedding_table.copyTo(tokens[t] * dim, state.x[t], 0, dim)
        );

        for (int l = 0; l < config.numberOfLayers; l++) {
            final int curLayer = l;
            Parallel.parallelFor(0, nTokens, t ->
                rmsnorm(state.xb[t], state.x[t], weights.rms_att_weight[curLayer], dim, config.rmsNormEps)
            );

            weights.wq[l].matmul(nTokens, state.xb, state.q, dim, dim);
            weights.wk[l].matmul(nTokens, state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(nTokens, state.xb, state.v, kvDim, dim);

            Parallel.parallelFor(0, nTokens, t -> {
                for (int i = 0; i < dim; i += 2) {
                    int head_dim = i % headSize;
                    float fcr = weights.freq_cis_real.get((position + t) * (headSize / 2) + (head_dim / 2));
                    float fci = weights.freq_cis_imag.get((position + t) * (headSize / 2) + (head_dim / 2));
                    int rotn = i < kvDim ? 2 : 1;
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = vi == 0 ? state.q[t] : state.k[t];
                        float v0 = vec.getFloat(i);
                        float v1 = vec.getFloat(i + 1);
                        vec.setFloat(i, v0 * fcr - v1 * fci);
                        vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                    }
                }
            });

            Parallel.parallelFor(0, nTokens, t -> {
                state.k[t].copyTo(0, state.keyCache[curLayer], (position + t) * kvDim, kvDim);
                state.v[t].copyTo(0, state.valueCache[curLayer], (position + t) * kvDim, kvDim);
            });

            if (!computeLogits && curLayer == config.numberOfLayers - 1) {
                state.idxPrevBlock = nTokens - 1;
                return null;
            }

            Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
                int token = (int) (ht / config.numberOfHeads);
                int h = (int) (ht % config.numberOfHeads);
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength;
                for (int t = 0; t <= position + token; t++) {
                    int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    float score = state.q[token].dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    state.att[token].setFloat(attOffset + t, score);
                }

                state.att[token].softmaxInPlace(attOffset, position + token + 1);

                for (int t = 0; t <= position + token; t++) {
                    int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    float a = state.att[token].getFloat(attOffset + t);
                    state.q[token].saxpyInPlace(qOffset, state.valueCache[curLayer], keyCacheOffset, headSize, a);
                }

                state.q[token].copyTo(qOffset, state.xb[token], qOffset, headSize);
            });

            Parallel.parallelFor(0, nTokens, t -> {
                state.x[t].addInPlace(state.xb[t]);
            });
            weights.wo[l].matmul(nTokens, state.xb, state.x, dim, dim);

            Parallel.parallelFor(0, nTokens, t ->
                rmsnorm(state.xb[t], state.x[t], weights.rms_ffn_weight[curLayer], dim, config.rmsNormEps)
            );

            weights.w1[l].matmul(nTokens, state.xb, state.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(nTokens, state.xb, state.hb2, config.hiddenDim, dim);

            Parallel.parallelFor(0, nTokens, t -> {
                state.hb[t].mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            });

            Parallel.parallelFor(0, nTokens, t -> {
                state.hb[t].multiplyInPlace(state.hb2[t]);
            });

            weights.w2[l].matmul(nTokens, state.hb, state.xb, dim, config.hiddenDim);

            Parallel.parallelFor(0, nTokens, t -> {
                state.x[t].addInPlace(state.xb[t]);
            });
        }

        Parallel.parallelFor(0, nTokens, t -> {
            rmsnorm(state.x[t], state.x[t], weights.rms_final_weight, dim, config.rmsNormEps);
        });

        weights.wcls.matmul(state.x[nTokens - 1], state.logits, config.vocabularySize, dim);
        state.idxPrevBlock = nTokens - 1;

        return state.logits;
    }

    public static List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
                                               IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        long startGen = 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken;
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            if (promptIndex < promptTokens.size()) {
                final int nTokens = Math.min(maxTokens - position, Math.min(promptTokens.size() - promptIndex, state.batchsize));
                final int[] tokens = new int[nTokens];
                for (int i = 0; i < nTokens; i++) {
                    tokens[i] = promptTokens.get(promptIndex + i);
                    if (echo) {
                        System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(tokens[i]))));
                    }
                }
                if (echo) {
                    System.out.format("position=%d, promptIdx=%d, promptSize=%d, tokens=%s%n", position, promptIndex, promptTokens.size(), Arrays.toString(tokens));
                }
                boolean computeLogits = promptIndex + nTokens >= promptTokens.size();
                forward(model, state, tokens, position, computeLogits);
                position += nTokens - 1;
                promptIndex += nTokens;
                if (promptIndex < promptTokens.size()) {
                    continue;
                }
                startGen = System.nanoTime();
            } else {
                forward(model, state, new int[]{token}, position, true);
            }
            nextToken = sampler.sampleToken(state.logits);
            if (echo) {
                System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
            }
            generatedTokens.add(nextToken);
            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }
            if (stopTokens.contains(nextToken)) {
                break;
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        System.err.printf("%ncontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%n",
                startPosition + promptIndex + generatedTokens.size(), model.configuration().contextLength,
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(),
                generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size());

        return generatedTokens;
    }
}

