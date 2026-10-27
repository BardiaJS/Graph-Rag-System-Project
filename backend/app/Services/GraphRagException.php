<?php

namespace App\Services;

use RuntimeException;

/**
 * Thrown when the Python service or the language model cannot be reached or
 * returns an error. Controllers should catch this and answer 503/502.
 */
class GraphRagException extends RuntimeException
{
}
