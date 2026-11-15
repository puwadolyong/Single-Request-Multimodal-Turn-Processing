using System.Text.Json;
using Convia.LicenseServer;
using Npgsql;

// An admin command runs against the database and exits; anything else starts the server.
// Sharing one entry point means the commands cannot drift from the schema the server uses,
// which a separate tool with its own copy of the SQL eventually would.
try
{
    // OpenStore is passed rather than called: `help` needs no database, and starting the
    // server should not open one twice.
    int? adminExit = Admin.Run(args, OpenStore);
    if (adminExit is not null) return adminExit.Value;
}
catch (Exception e) when (e is NpgsqlException or InvalidOperationException)
{
    // A stack trace is the wrong answer to a mistyped connection string. Every one of these
    // is a configuration mistake with an obvious fix, and the fix is what should be on screen.
    Console.Error.WriteLine(ExplainDatabaseFailure(e));
    return 2;
}

// `dashboard` is the local-only way in: a throwaway token, a loopback address, and the
// browser opened on it. Administration on the deployed server is meant to stay switched off,
// and asking somebody to invent a 24-character secret to manage licences on their own laptop
// is ceremony that protects nothing - the port is not reachable from anywhere else.
bool localDashboard = args.Length > 0 &&
                      args[0].Equals("dashboard", StringComparison.OrdinalIgnoreCase);
string? sessionToken = null;

// The content root decides where wwwroot is looked for, and it defaults to the working
// directory - so the dashboard page vanished whenever the command was run from anywhere but
// the folder the program lives in, which for a command on the PATH is almost always. Pinned
// to the program's own folder, where its files actually are.
WebApplicationBuilder builder = WebApplication.CreateBuilder(new WebApplicationOptions
{
    Args = args,
    ContentRootPath = AppContext.BaseDirectory
});

if (localDashboard)
{
    sessionToken = Admin.GenerateKey() + Admin.GenerateKey();   // well past the minimum
    Environment.SetEnvironmentVariable("CONVIA_ADMIN_TOKEN", sessionToken);

    // Loopback only. Binding to every interface would put licence administration on the
    // office network for as long as the window is open.
    //
    // A fixed port gives a stable address, but a tool people open and close all day will
    // eventually meet one already in use - a window left open, or one still shutting down.
    // Forty lines of Kestrel stack trace is the wrong answer to that, so a taken port simply
    // means the next free one.
    builder.WebHost.UseUrls($"http://127.0.0.1:{FreePort(8765)}");
}

// Which websites a browser may call the status endpoint from.
//
// Only that endpoint needs this, and only because the customer portal runs in the browser.
// The application itself is not a browser and never sends an Origin, so CORS has no bearing
// on it - which is why none of this touches /v1/turn.
//
// Named origins rather than any: allowing any would let a page on any site read a licence
// out of a visitor's browser if it could guess a key, and would do so without the visitor
// seeing anything happen.
string[] portalOrigins = (Environment.GetEnvironmentVariable("CONVIA_PORTAL_ORIGINS") ?? "")
    .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

const string PortalCorsPolicy = "portal";
builder.Services.AddCors(options => options.AddPolicy(PortalCorsPolicy, policy =>
{
    // No origins configured means no browser may call it. The default has to be the closed
    // one: a wide-open default is invisible until somebody looks for it.
    if (portalOrigins.Length == 0) return;

    policy.WithOrigins(portalOrigins)
          .WithMethods("GET")
          .WithHeaders("Authorization", "Content-Type");
}));

builder.Services.AddSingleton(_ => OpenStore());
builder.Services.AddHttpClient<GeminiProxy>(c => c.Timeout = GeminiProxy.MaxUpstreamTimeout);
builder.Services.AddSingleton(_ => ResolveGeminiKey());

// The longest voice turn, the two-minute IELTS long turn, carries ~5 MB of base64 audio;
// anything beyond this is not a learner speaking.
builder.WebHost.ConfigureKestrel(o => o.Limits.MaxRequestBodySize = 8 * 1024 * 1024);

// Cloud Run, and every other container host of its kind, assigns the port at run time and
// expects the process to listen on all interfaces. ASP.NET Core does not read PORT on its
// own; without this the container starts, listens on its own default, answers nothing, and
// is killed by a startup probe whose message says only that the container failed to listen.
// Absent locally, where the address stays whatever the command line gave.
string? assignedPort = Environment.GetEnvironmentVariable("PORT");
if (!string.IsNullOrWhiteSpace(assignedPort))
    builder.WebHost.UseUrls($"http://0.0.0.0:{assignedPort.Trim()}");

WebApplication app = builder.Build();

LicenseStore store = app.Services.GetRequiredService<LicenseStore>();
app.Logger.LogInformation("Licences in {Database}", store.DatabaseDescription);

app.Logger.LogInformation(
    portalOrigins.Length == 0
        ? "No browser origins allowed. Set CONVIA_PORTAL_ORIGINS to let a website read "
          + "/v1/license/status from a browser."
        : "Browsers may call /v1/license/status from: {Origins}",
    string.Join(", ", portalOrigins));

string portalToken = (Environment.GetEnvironmentVariable("CONVIA_PORTAL_TOKEN") ?? "").Trim();
app.Logger.LogInformation(
    PortalApi.IsEnabled(portalToken)
        ? "Portal lookup is on at /v1/licences/by-customer."
        : "Portal lookup is OFF. Set CONVIA_PORTAL_TOKEN to at least {Minimum} characters "
          + "to turn it on; until then that route answers 404.",
    AdminApi.MinimumTokenLength);

string adminToken = ResolveAdminToken();
app.Logger.LogInformation(
    AdminApi.IsEnabled(adminToken)
        ? "Licence administration is on at /admin."
        : "Licence administration is OFF. Set CONVIA_ADMIN_TOKEN to at least "
          + "{Minimum} characters to turn it on; until then /admin and its API answer 404.",
    AdminApi.MinimumTokenLength);

string key = app.Services.GetRequiredService<string>();
app.Logger.LogInformation(
    string.IsNullOrWhiteSpace(key)
        ? "No Gemini key found. Set CONVIA_GEMINI_KEY or put the key in gemini.key at the "
          + "repository root. /v1/turn will refuse every request until then."
        : "Gemini key loaded ({Length} chars).", key.Length);

// Said at startup because it is the setting that decides what every turn costs, and a
// switch that did not take is otherwise only discovered on the bill.
if (GeminiProxy.ModelWasRejected)
    app.Logger.LogWarning(
        "CONVIA_GEMINI_MODEL is set to something that is not a model name, so it was ignored. "
        + "Using {Model}.", GeminiProxy.Model);
else
    app.Logger.LogInformation("Gemini model: {Model}.", GeminiProxy.Model);

// Said whenever it is not the default, so a raised limit meant for a local batch run
// cannot sit unnoticed on the deployed service.
if (LicenseStore.MaxTurnsPerMinute != LicenseStore.DefaultMaxTurnsPerMinute)
    app.Logger.LogWarning("Rate limit raised to {Limit} turns per licence per minute (default {Default}).",
        LicenseStore.MaxTurnsPerMinute, LicenseStore.DefaultMaxTurnsPerMinute);

// ──────────────────────────── helpers ────────────────────────────

/// <summary>
///     Where the operator's Gemini key comes from.
///
///     <para>
///     The environment variable is what a deployment sets. The file exists because the
///     alternative during development was reading the key out of the Unity scene, which is
///     exactly where it should not live - and once it was removed from there, pasting it into
///     a terminal on every restart was the only thing left. It sits at the repository root
///     rather than inside the server folder, and is gitignored.
///     </para>
/// </summary>
static string ResolveGeminiKey()
{
    string? fromEnv = Environment.GetEnvironmentVariable("CONVIA_GEMINI_KEY");
    if (!string.IsNullOrWhiteSpace(fromEnv)) return fromEnv.Trim();

    try
    {
        // server/Convia.LicenseServer -> repository root
        string path = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "gemini.key");
        if (File.Exists(path)) return File.ReadAllText(path).Trim();
    }
    catch (Exception)
    {
        // Falls through to empty, which /v1/turn reports as not_configured.
    }

    return "";
}

/// <summary>
///     Connects to the licence database.
///
///     <para>
///     The sample licences are seeded only when asked for by name. Their keys are printed in
///     the README, so seeding them wherever the server happens to run would put a working
///     licence in the hands of everyone who can read it. Inferring "this looks like
///     development" from the host or the environment name was the alternative and is the kind
///     of guess that is wrong exactly once.
///     </para>
/// </summary>
static LicenseStore OpenStore() =>
    new(ResolveConnectionString(),
        seedDevelopmentLicences: Environment.GetEnvironmentVariable("CONVIA_SEED_DEV_LICENCES")
            is "1" or "true" or "TRUE");

/// <summary>
///     Says what to do about a database that would not open.
///
///     <para>
///     The wrong-password case names the trap it comes from: a provider's console shows the
///     password masked, and copying before revealing it yields a string that looks complete
///     and is not. That cost an afternoon once already.
///     </para>
/// </summary>
static string ExplainDatabaseFailure(Exception e)
{
    if (e is NpgsqlException { SqlState: "28P01" })
    {
        return """
            The database refused those credentials.
            Check the password in CONVIA_DB_CONNECTION. If it was copied from a hosting
            provider's console, reveal the password there before copying: a masked string
            copies as asterisks or question marks and looks complete.
            """;
    }

    if (e is NpgsqlException { SqlState: "3D000" })
        return "That database does not exist. Check the name at the end of CONVIA_DB_CONNECTION.";

    if (e is InvalidOperationException) return e.Message;

    return $"""
        Could not reach the database: {e.Message}
        Check CONVIA_DB_CONNECTION, and that the host is reachable from here.
        """;
}

/// <summary>
///     The administration token, from CONVIA_ADMIN_TOKEN.
///
///     <para>
///     Environment only - no file fallback, unlike the database string and the model key. Those
///     two are read from disk because the commands that need them run on the operator's own
///     machine; this one is only ever wanted by the deployed server, where it arrives from a
///     secret store. A file fallback would be somewhere else for it to be left lying around.
///     </para>
/// </summary>
/// <summary>
///     The preferred port when it is free, otherwise 0, which asks the operating system for
///     any free one.
/// </summary>
static int FreePort(int preferred)
{
    try
    {
        var probe = new System.Net.Sockets.TcpListener(
            System.Net.IPAddress.Loopback, preferred);
        probe.Start();
        probe.Stop();
        return preferred;
    }
    catch (System.Net.Sockets.SocketException)
    {
        return 0;
    }
}

static string ResolveAdminToken() =>
    (Environment.GetEnvironmentVariable("CONVIA_ADMIN_TOKEN") ?? "").Trim();

/// <summary>The database to talk to, from CONVIA_DB_CONNECTION.</summary>
static string ResolveConnectionString()
{
    string raw = ConnectionStrings.Configured();

    if (raw.Length == 0)
    {
        throw new InvalidOperationException($"""
            No database configured. Either set CONVIA_DB_CONNECTION, or put the connection
            string in this file:
              {ConnectionStrings.FilePath}
            Either the postgresql://user:password@host/database form or the
            Host=...;Username=...;Password=...;Database=... form will do.
            """);
    }

    return ConnectionStrings.Normalise(raw);
}

/// <summary>
///     One licence as the outside world sees it.
///
///     <para>
///     Shared by both read endpoints on purpose. Two places building this by hand is how the
///     account page and the portal end up disagreeing about what somebody owns.
///     </para>
/// </summary>
static object Describe(License license, LicenseStore store) => new
{
    key = license.Key,
    plan = license.Plan,
    customer = license.Customer,
    expiresAt = license.ExpiresAt,
    revoked = license.Revoked,

    // Remaining is computed here rather than left to the caller, so a page cannot show a
    // different number from the one that decides whether the next turn is allowed.
    credits = new
    {
        total = license.QuotaTurns,
        used = license.UsedTurns,
        remaining = license.TurnsRemaining
    },

    seats = new
    {
        used = license.SeatsUsed,
        allowed = license.SeatsAllowed,
        devices = store.Seats(license.Key).Select(s => new { name = s.DeviceName, lastSeen = s.LastSeen })
    },

    status = license.Revoked ? "revoked"
        : license.ExpiresAt < DateTimeOffset.UtcNow ? "expired"
        : license.TurnsRemaining == 0 ? "empty"
        : "active"
};

/// <summary>
///     Renders a refusal.
///
///     <para>
///     The plan travels with it for two reasons. It picks the wording here - a prepaid
///     balance does not come back next month - and it goes into the body so the application
///     can word its own follow-up line, which it otherwise cannot: the plan is only ever
///     reported on a successful verify, so at the exact moment a customer is refused for
///     running out, the client knows least about what they bought.
///     </para>
/// </summary>
static IResult Fail(string code, string? plan = null)
{
    (int status, string message) = ErrorCodes.Describe(code, plan);
    return Results.Json(
        new { error = new { code, message }, plan },
        statusCode: status);
}

/// <summary>
///     Shown back to the customer on the account page so they can tell seats apart.
///
///     <para>
///     Read from the verify body, which is where the contract puts it and where the client
///     sends it. An earlier version of this read an X-Device-Name header that nothing ever
///     sent, so every seat was recorded nameless.
///     </para>
/// </summary>
static async Task<string?> ReadDeviceNameAsync(HttpRequest r, CancellationToken ct)
{
    try
    {
        using JsonDocument doc = await JsonDocument.ParseAsync(r.Body, cancellationToken: ct);
        if (doc.RootElement.TryGetProperty("deviceName", out JsonElement name) &&
            name.ValueKind == JsonValueKind.String)
        {
            string value = name.GetString() ?? "";
            return string.IsNullOrWhiteSpace(value) ? null : value;
        }
    }
    catch (JsonException)
    {
        // An empty or unparseable body is fine; the seat is simply unnamed.
    }

    return null;
}

static string DeviceId(HttpRequest r) =>
    r.Headers.TryGetValue("X-Device-Id", out var v) && !string.IsNullOrWhiteSpace(v)
        ? v.ToString()
        : "unknown-device";

static string? BearerKey(HttpRequest r)
{
    string? header = r.Headers.Authorization.ToString();
    if (string.IsNullOrWhiteSpace(header)) return null;
    const string prefix = "Bearer ";
    return header.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)
        ? header[prefix.Length..].Trim()
        : header.Trim();
}

static void WriteQuotaHeaders(HttpResponse response, License license)
{
    response.Headers["X-Quota-Limit"] = license.QuotaTurns.ToString();
    response.Headers["X-Quota-Used"] = license.UsedTurns.ToString();
    response.Headers["X-Quota-Resets"] = license.ResetsAt.ToString("O");
}

// ──────────────────────────── endpoints ────────────────────────────

app.UseCors();

app.MapGet("/health", () => Results.Ok(new { ok = true }));

// The dashboard is a plain page with no secret in it; it asks for the token and keeps it in
// the browser. Serving it from here means one deployment rather than two, and no question
// about which origin is allowed to call the API.
app.UseStaticFiles();
app.MapGet("/admin", () => Results.Redirect("/admin.html"));

AdminApi.Map(app, adminToken);

app.MapPost("/v1/license/verify", async (HttpRequest req, LicenseStore store,
                                         CancellationToken ct) =>
{
    string device = DeviceId(req);
    License? license = store.Find(BearerKey(req));

    string? refusal = store.Refuse(license, device);
    if (refusal is not null) return Fail(refusal, license?.Plan);

    // Claiming the seat here rather than on the first turn means a customer who opens the
    // application on a third machine is told so immediately, not after speaking a sentence.
    store.ClaimSeat(license!.Key, device, await ReadDeviceNameAsync(req, ct));

    // Re-read so the seat just taken is counted in what is reported back.
    license = store.Find(license.Key)!;

    return Results.Ok(new
    {
        status = "active",
        plan = license.Plan,
        expiresAt = license.ExpiresAt,
        quota = new
        {
            limitTurns = license.QuotaTurns,
            usedTurns = license.UsedTurns,
            resetsAt = license.ResetsAt
        },
        seats = new { used = license.SeatsUsed, allowed = license.SeatsAllowed }
    });
});

/// A read-only look at a licence, for the customer's own account page.
///
/// Separate from verify because verify claims a seat. A web page that called verify every
/// time somebody opened it would spend the customer's seats on their behalf, and they would
/// find themselves locked out of their own second machine by the page that exists to reassure
/// them. Nothing here writes.
///
/// The key is its own authorisation, as everywhere else in this server: knowing one already
/// lets you spend it, so being told what is left on it gives nothing further away.
app.MapGet("/v1/license/status", (HttpRequest req, LicenseStore store) =>
{
    License? license = store.Find(BearerKey(req));
    return license is null ? Fail(ErrorCodes.InvalidKey) : Results.Ok(Describe(license, store));
}).RequireCors(PortalCorsPolicy);

/// Every licence belonging to one customer, for the portal, which knows who is signed in but
/// not what they bought.
///
/// This one answers a question about somebody else - "what does this address own" - so unlike
/// the rest of the server it cannot be authorised by the licence key, and an email address is
/// not a secret. It carries a shared token instead, held by the portal and nowhere else.
/// Without that token configured the route does not exist, the same way administration does
/// not.
///
/// The portal must send only an address it has itself verified. Nothing here can tell the
/// difference between one somebody signed in with and one they typed into a box.
app.MapGet("/v1/licences/by-customer", (HttpRequest req, LicenseStore store) =>
{
    if (!PortalApi.IsAuthorised(req, portalToken)) return Results.NotFound();

    string email = (req.Query["email"].ToString() ?? "").Trim();
    if (email.Length == 0) return Fail(ErrorCodes.BadRequest);

    // An address with nothing against it is an answer, not a failure: it is what the portal
    // shows somebody who has signed up and not yet bought anything.
    return Results.Ok(new
    {
        email,
        licences = store.ForCustomer(email).Select(l => Describe(l, store))
    });
});

app.MapPost("/v1/license/release", (HttpRequest req, LicenseStore store) =>
{
    License? license = store.Find(BearerKey(req));
    if (license is null) return Fail(ErrorCodes.InvalidKey);

    store.ReleaseSeat(license.Key, DeviceId(req));

    License after = store.Find(license.Key)!;
    return Results.Ok(new { seats = new { used = after.SeatsUsed, allowed = after.SeatsAllowed } });
});

app.MapPost("/v1/turn", async (HttpRequest req, HttpResponse res,
                               LicenseStore store, GeminiProxy proxy,
                               CancellationToken ct) =>
{
    string device = DeviceId(req);
    License? license = store.Find(BearerKey(req));

    string? refusal = store.Refuse(license, device);
    if (refusal is not null)
    {
        if (license is not null) WriteQuotaHeaders(res, license);
        return Fail(refusal, license?.Plan);
    }

    // Before the body is even read, so that a client hammering this with rubbish is slowed
    // by the same rule as one hammering it with real turns - and so a refused attempt costs
    // nothing upstream.
    if (store.RegisterTurnAttempt(license!.Key, device) > LicenseStore.MaxTurnsPerMinute)
    {
        res.Headers["Retry-After"] = LicenseStore.SecondsUntilNextWindow().ToString();
        WriteQuotaHeaders(res, license);
        return Fail(ErrorCodes.RateLimited);
    }

    // No name here: the turn body is the turn. A seat first seen on a turn stays unnamed
    // until the next verify, which is every launch.
    store.ClaimSeat(license.Key, device, null);

    TurnRequest? turn;
    try
    {
        turn = await JsonSerializer.DeserializeAsync<TurnRequest>(
            req.Body,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true },
            ct);
    }
    catch (JsonException)
    {
        return Fail(ErrorCodes.BadRequest);
    }

    if (turn is null) return Fail(ErrorCodes.BadRequest);

    if (turn.Mode == "voice")
    {
        if (turn.Audio?.Data is null) return Fail(ErrorCodes.BadRequest);
        if (turn.Audio.Data.Length > GeminiProxy.MaxAudioBase64Chars) return Fail(ErrorCodes.AudioTooLarge);
    }
    else if (string.IsNullOrWhiteSpace(turn.Text))
    {
        return Fail(ErrorCodes.BadRequest);
    }

    GeminiProxy.Result result = await proxy.SendTurnAsync(turn, ct);

    if (!result.Ok)
    {
        // Not counted. A turn the learner never received is not a turn they spent.
        WriteQuotaHeaders(res, license!);
        return Fail(result.ErrorCode ?? ErrorCodes.UpstreamError);
    }

    // Counted once the tokens are actually spent, whatever the payload turns out to contain.
    // A malformed response still cost money, and the client's enforcement layer deals with it.
    WriteQuotaHeaders(res, store.CountTurn(license.Key) ?? license);

    return Results.Content(result.Body, "application/json");
});

if (localDashboard)
{
    // The token travels in the fragment, which browsers do not send to the server and which
    // the page clears once it has read it. Putting it in the query string would write it into
    // the server's own request log.
    app.Start();

    // Read back rather than assumed: when the preferred port was taken the real one was
    // chosen by the operating system, and printing the wrong address would be worse than
    // printing none.
    string origin = app.Urls.FirstOrDefault() ?? "http://127.0.0.1:8765";
    string url = $"{origin.TrimEnd('/')}/admin.html#t={Uri.EscapeDataString(sessionToken!)}";
    Console.WriteLine();
    Console.WriteLine("  Licence dashboard: " + url);
    Console.WriteLine("  Press Ctrl+C to close it.");
    Console.WriteLine();

    try
    {
        System.Diagnostics.Process.Start(
            new System.Diagnostics.ProcessStartInfo(url) { UseShellExecute = true });
    }
    catch (Exception)
    {
        // No browser to launch, or no shell to launch it with. The address is on screen.
    }

    app.WaitForShutdown();
    return 0;
}

app.Run();
return 0;
