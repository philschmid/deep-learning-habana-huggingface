#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { EcsClusterStack } from '../lib/ecs_stack';

const app = new cdk.App();
new EcsClusterStack(app, 'EcsClusterStack', {});